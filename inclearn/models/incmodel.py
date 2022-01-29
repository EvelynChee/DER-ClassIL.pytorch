import numpy as np
import collections
import random
import time
import math
import os
from copy import deepcopy
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.autograd import Variable

from inclearn.convnet import network
from inclearn.models.base import IncrementalLearner
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter
from inclearn.tools.memory import MemorySize
from inclearn.tools.scheduler import GradualWarmupScheduler
from inclearn.convnet.utils import extract_features, update_classes_mean, finetune_last_layer

# Constants
EPSILON = 1e-8

class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=8):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.marginloss = nn.MarginRankingLoss(margin=margin)
        self.num_instances = num_instances
                
    def __call__(self, inputs, targets):    
        n = inputs.size(0)
        
        gt_index = torch.zeros(inputs.size()).cuda()
        gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
        gt_scores = inputs.masked_select(gt_index)
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        inputs_tile = inputs.repeat(n, 1).view(n,n,-1)
        gt_index_tile = gt_index.t().repeat(n,1).t().view(n,n,-1)
        gt_scores_tile = (inputs_tile * gt_index_tile).sum(axis=2)
        
        max1 = gt_scores_tile.max(axis=1, keepdims=True)[0]
        tmp1 = ((gt_scores_tile - max1) * mask).topk(k=min(n, self.num_instances), axis=1, largest=False)[0] + max1

        min2 = gt_scores_tile.min(axis=1, keepdims=True)[0]
        tmp2 = ((gt_scores_tile - min2) * ~mask).topk(k=min(n, self.num_instances), axis=1)[0] + min2

        tmp1 = (gt_scores.view(-1,1) - tmp1).abs()
        tmp2 = gt_scores.view(-1,1) - tmp2

        dist_ap = tmp1.view(n,-1,1).repeat(1,1,self.num_instances).view(n,-1).view(-1)
        dist_an = tmp2.repeat(1,self.num_instances).view(n,-1).view(-1)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.marginloss(dist_an, dist_ap, y)
            
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss
        
        
class IncModel(IncrementalLearner):
    def __init__(self, cfg, trial_i, _run, ex, tensorboard, inc_dataset):
        super().__init__()
        self._cfg = cfg
        self._device = cfg['device']
        self._ex = ex
        self._run = _run  # the sacred _run object.

        # Data
        self._inc_dataset = inc_dataset
        self._n_classes = 0
        self._trial_i = trial_i  # which class order is used

        # Optimizer paras
        self._opt_name = cfg["optimizer"]
        self._warmup = cfg['warmup']
        self._lr = cfg["lr"]
        self._weight_decay = cfg["weight_decay"]
        self._n_epochs = cfg["epochs"]
        self._scheduling = cfg["scheduling"]
        self._lr_decay = cfg["lr_decay"]

        # Classifier Learning Stage
        self._decouple = cfg["decouple"]

        # Logging
        self._tensorboard = tensorboard
        if f"trial{self._trial_i}" not in self._run.info:
            self._run.info[f"trial{self._trial_i}"] = {}
        self._val_per_n_epoch = cfg["val_per_n_epoch"]

        # Model
        self._der = cfg['der']  # Whether to expand the representation
        self._network = network.BasicNet(
            cfg["convnet"],
            cfg=cfg,
            nf=cfg["channel"],
            device=self._device,
            use_bias=cfg["use_bias"],
            dataset=cfg["dataset"],
        )
        self._parallel_network = DataParallel(self._network)
        self._train_head = cfg["train_head"]
        self._infer_head = cfg["infer_head"]
        self._old_model = None

        # Learning
        self._dist_loss = cfg["distillation_loss"]
        self._rank_loss = cfg["ranking_loss"]
        self._clf_loss = cfg["classification_loss"]

        # Memory
        self._memory_size = MemorySize(cfg["mem_size_mode"], inc_dataset, cfg["memory_size"],
                                       cfg["fixed_memory_per_cls"])
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]

        if self._cfg["save_ckpt"]:
            save_path = os.path.join(os.getcwd(), "ckpts", cfg["exp"]["name"])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts", cfg["exp"]["name"], "mem")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

    def eval(self):
        self._parallel_network.eval()
            
    def train(self):
        if self._der:
            self._parallel_network.train()
            self._parallel_network.module.convnets[-1].train()
            if self._task >= 1:
                for i in range(self._task):
                    self._parallel_network.module.convnets[i].eval()
        else:
            self._parallel_network.train()
            
    def _before_task(self, taski, inc_dataset):
        self._ex.logger.info(f"Begin step {taski}")
        
        # Update Task info
        self._task = taski
        self.the_lambda = 5.0 * math.sqrt(self._n_classes/self._task_size)
        self._n_classes += self._task_size

        # Memory
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        self._ex.logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self._network.add_classes(self._task_size)
        self._network.task_size = self._task_size
        self.set_optimizer()

        samples_per_cls = np.zeros(self._n_classes)
        for i in range(self._n_classes):
            samples_per_cls[i] = np.sum(inc_dataset.targets_inc==i)
        self._ex.logger.info("Now {} training samples per class.".format(samples_per_cls))

        effective_num = 1.0 - np.power(self._clf_loss['beta'], samples_per_cls)
        weights = (1.0 - self._clf_loss['beta']) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)
        self._clf_loss['weights'] = torch.tensor(weights).float().cuda()            
        
#         self._ex.logger.info("Ratio: {}".format(self._clf_loss['weights'].max()/self._clf_loss['weights'].min()))
                        
    def set_optimizer(self, lr=None):
        if lr is None:
            lr = self._lr

        if self._cfg["dynamic_weight_decay"]:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self._cfg["task_max"] / (self._task + 1)
        else:
            weight_decay = self._weight_decay
        self._ex.logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))

#         if self._der and self._task > 0:
#             for i in range(self._task):
#                 for p in self._parallel_network.module.convnets[i].parameters():
#                     p.requires_grad = False

# #             for p in self._parallel_network.classifier.parameters():
# #                 p.requires_grad = False
                    
#         self._optimizer = factory.get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()),
#                                                 self._opt_name, lr, weight_decay)
        
        params = []
        if self._der and self._task > 0:
            for i in range(self._task):
                for p in self._parallel_network.module.convnets[i].parameters():
                    p.requires_grad = False
#                 params.append({'params': self._parallel_network.module.convnets[i].parameters(),
#                                'lr': lr * 0.001, 'weight_decay': weight_decay})

        self.params1 = []
        self.params2 = []
        for n, p in self._parallel_network.module.convnets[-1].named_parameters():
            if 'alpha' in n or 'merge' in n:
                self.params1.append(p)
            else:
                self.params2.append(p)
        
        self.params2 += list(self._parallel_network.module.classifier.parameters()) + list(self._parallel_network.module.aux_classifier.parameters())
                
        self._optimizer1 = torch.optim.SGD(self.params1, lr=lr, weight_decay=weight_decay, momentum=0.9)        
        self._optimizer = torch.optim.SGD(self.params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        
#         params.append({'params': self._parallel_network.module.convnets[-1].parameters(),
#                        'lr': lr, 'weight_decay': weight_decay})
#         params.append({'params': self._parallel_network.module.classifier.parameters(),
#                        'lr': lr*0.1, 'weight_decay': weight_decay})
#         params.append({'params': self._parallel_network.module.aux_classifier.parameters(),
#                        'lr': lr, 'weight_decay': weight_decay})
                
#         self._optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
               

        if "cos" in self._cfg["scheduler"]:
            self._scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer1, self._n_epochs)
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self._n_epochs)
        else:
            self._scheduler1 = torch.optim.lr_scheduler.MultiStepLR(self._optimizer1,
                                                                   self._scheduling,
                                                                   gamma=self._lr_decay)
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   self._scheduling,
                                                                   gamma=self._lr_decay)

        if self._warmup:
            print("warmup")
            self._warmup_scheduler1 = GradualWarmupScheduler(self._optimizer1,
                                                            multiplier=1,
                                                            total_epoch=self._cfg['warmup_epochs'],
                                                            after_scheduler=self._scheduler)            
            self._warmup_scheduler = GradualWarmupScheduler(self._optimizer,
                                                            multiplier=1,
                                                            total_epoch=self._cfg['warmup_epochs'],
                                                            after_scheduler=self._scheduler)            

#     def _train_task(self, train_loader, val_loader):
#         self._ex.logger.info(f"nb {len(train_loader.dataset)}")

#         topk = 5 if self._n_classes > 5 else self._task_size
#         accu = ClassErrorMeter(accuracy=True, topk=[1, topk])
#         train_new_accu = ClassErrorMeter(accuracy=True)
#         train_old_accu = ClassErrorMeter(accuracy=True)

#         utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "Initial trainset")
#         utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
#                                    self._increments, "Initial trainset")

#         self._optimizer.zero_grad()
#         self._optimizer.step()

#         for epoch in range(self._n_epochs):                
#             self._metrics = collections.defaultdict(float)
            
#             accu.reset()
#             train_new_accu.reset()
#             train_old_accu.reset()
#             if self._warmup:
#                 self._warmup_scheduler.step()
                
#             for i, (inputs, targets) in enumerate(train_loader, start=1):
#                 self.train()
#                 self._optimizer.zero_grad()
#                 old_classes = targets < (self._n_classes - self._task_size)
#                 new_classes = targets >= (self._n_classes - self._task_size)
#                 loss = self._forward_loss(
#                     epoch,
#                     inputs,
#                     targets,
#                     old_classes,
#                     new_classes,
#                     accu=accu,
#                     new_accu=train_new_accu,
#                     old_accu=train_old_accu,
#                 )

#                 if not utils.check_loss(loss):
#                     import pdb
#                     pdb.set_trace()
                              
#                 loss.backward()
                        
# #                 # Set fixed param grads to 0.
# #                 if epoch < self._clf_loss['warmup']:
# #                     self._parallel_network.module.classifier.weight.grad.data[:-self._inc_dataset.increments[self._task],:] = 0
# #                     self._parallel_network.module.classifier.weight.grad.data[-self._inc_dataset.increments[self._task]:, -self._network.out_dim:] *= 0.01
# #                     self._parallel_network.module.classifier.weight.grad.data[:-self._inc_dataset.increments[self._task], :] *= 0.01
# #                     self._parallel_network.module.classifier.sigma.grad.data.add_(self._parallel_network.module.classifier.sigma.data, alpha=self._weight_decay) 
                    
#                 self._optimizer.step()

#                 if self._cfg["postprocessor"]["enable"]:
#                     if self._cfg["postprocessor"]["type"].lower() == "wa":
#                         for p in self._network.classifier.parameters():
#                             p.data.clamp_(0.0)

#             if not self._warmup:
#                 self._scheduler.step()
                
#             self._print_metrics(epoch, i, accu)
            
#             if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
#                 self.validate(val_loader)
                            

#         # For the large-scale dataset, we manage the data in the shared memory.
#         self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory

#         utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "After training")
#         utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
#                                    self._increments, "Trainset")
#         self._run.info[f"trial{self._trial_i}"][f"task{self._task}_train_accu"] = round(accu.value()[0], 3)
        
    def _train_task(self, train_loader, val_loader):
        self._ex.logger.info(f"nb {len(train_loader.dataset)}")

        topk = 5 if self._n_classes > 5 else self._task_size
        accu = ClassErrorMeter(accuracy=True, topk=[1, topk])
        train_new_accu = ClassErrorMeter(accuracy=True)
        train_old_accu = ClassErrorMeter(accuracy=True)

        utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "Initial trainset")
        utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Initial trainset")

        self._optimizer.zero_grad()
        self._optimizer.step()

        for epoch in range(self._n_epochs):
            self._metrics = collections.defaultdict(float)
            
            accu.reset()
            train_new_accu.reset()
            train_old_accu.reset()
            if self._warmup:
                self._warmup_scheduler.step()
                self._warmup_scheduler1.step()
                if epoch == self._cfg['warmup_epochs']:
                    self._network.classifier.reset_parameters()
                    if self._cfg['use_aux_cls']:
                        self._network.aux_classifier.reset_parameters()

#             if epoch > 0 and self._task > 0:
#                 self._parallel_network.module.classifier.weight.data[:-self._task_size, -self._network.out_dim:] = deepcopy(self._parallel_network.module.aux_classifier.weight.data[:1,-self._network.out_dim:])

            for i, (inputs, targets) in enumerate(train_loader, start=1):
                if i % 5 == 0:
                    for p in self.params1:
                        p.requires_grad = True
                    for p in self.params2:
                        p.requires_grad = False
                    self.eval()
                    self._optimizer1.zero_grad()
                else:
                    for p in self.params1:
                        p.requires_grad = False
                    for p in self.params2:
                        p.requires_grad = True
                    self.train()
                    self._optimizer.zero_grad()
        
#                 self.train()
#                 self._optimizer.zero_grad()
                old_classes = targets < (self._n_classes - self._task_size)
                new_classes = targets >= (self._n_classes - self._task_size)
                loss = self._forward_loss(
                    epoch,
                    inputs,
                    targets,
                    old_classes,
                    new_classes,
                    accu=accu,
                    new_accu=train_new_accu,
                    old_accu=train_old_accu,
                )

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                
                
#                 if epoch < 100:
#                     self._parallel_network.module.classifier.weight.grad.data[:] = 0.0    
                    
#                 self._parallel_network.module.classifier.weight.grad.data.add_(self._parallel_network.module.classifier.weight.data, alpha=self._weight_decay)
#                 self._parallel_network.module.classifier.sigma.grad.data.add_(self._parallel_network.module.classifier.sigma.data, alpha=self._weight_decay)                    
    
#                 # Set fixed param grads to 0.
#                 if epoch < 110 and self._task > 0:                    
#                     self._parallel_network.module.classifier.weight.grad.data[:-self._task_size, :] = 0.0
#                     self._parallel_network.module.classifier.sigma.grad.data[:] = 0.0
#                                 self._parallel_network.module.classifier.weight.grad.data.add_(self._parallel_network.module.classifier.weight.data, alpha=self._weight_decay) 
#                 self._parallel_network.module.classifier.weight.grad.data[:-self._inc_dataset.increments[self._task], :-self._network.out_dim] *= 0.001
#                 self._parallel_network.module.classifier.weight.grad.data[:-self._inc_dataset.increments[self._task], -self._network.out_dim:] *= 0.01
#                 self._parallel_network.module.classifier.sigma.grad.data.add_(self._parallel_network.module.classifier.sigma.data, alpha=self._weight_decay) 
                
                if i % 5 == 0:
                    self._optimizer1.step()
                else:
                    self._optimizer.step()
                    
#                 self._optimizer.step()

                if self._cfg["postprocessor"]["enable"]:
                    if self._cfg["postprocessor"]["type"].lower() == "wa":
                        for p in self._network.classifier.parameters():
                            p.data.clamp_(0.0)
            
            if not self._warmup:
                self._scheduler1.step()
                self._scheduler.step()
                
            self._print_metrics(epoch, i, accu)
            
            if self._val_per_n_epoch > 0 and (epoch+1) % self._val_per_n_epoch == 0:
                self.validate(val_loader)

                        
        # For the large-scale dataset, we manage the data in the shared memory.
        self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory

        utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "After training")
        utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Trainset")
        self._run.info[f"trial{self._trial_i}"][f"task{self._task}_train_accu"] = round(accu.value()[0], 3)

    def _print_metrics(self, epoch, nb_batches, accu):
        pretty_metrics = ", ".join(
            "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
            for metric_name, metric_value in self._metrics.items()
        )
        
        pretty_metrics += ", Train Accu@1: {}, Train Acc@5: {}".format(round(accu.value()[0], 3), round(accu.value()[1], 3),)
        
        self._ex.logger.info(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, self._n_epochs, pretty_metrics
            )
        )
        
    def _forward_loss(self, epoch, inputs, targets, old_classes, new_classes, accu=None, new_accu=None, old_accu=None):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

        outputs = self._parallel_network(inputs)
        if accu is not None:
            accu.add(outputs['logit'], targets)
            # accu.add(logits.detach(), targets.cpu().numpy())
        # if new_accu is not None:
        #     new_accu.add(logits[new_classes].detach(), targets[new_classes].cpu().numpy())
        # if old_accu is not None:
        #     old_accu.add(logits[old_classes].detach(), targets[old_classes].cpu().numpy())
        return self._compute_loss(epoch, inputs, targets, outputs, old_classes, new_classes)

    def _compute_loss(self, epoch, inputs, targets, outputs, old_classes, new_classes):
        num_old_classes = sum(self._inc_dataset.increments[:self._task])
                
        if self._clf_loss['warmup'] < 0 or epoch < self._clf_loss['warmup'] or self._task == 0:
            loss = F.cross_entropy(outputs['logit'], targets)
#             loss = torch.tensor(0.0).cuda()
        else:
            loss = FocalLoss(gamma=self._clf_loss['gamma'], alpha=self._clf_loss['weights'])(outputs['logit'], targets) 
            if torch.nonzero(new_classes).size(0) > 0:
                loss += TripletLoss(margin=self._clf_loss['margin'])(outputs['logit_bs'][new_classes], targets[new_classes])
        
        self._metrics["Clf loss"] += loss.item()               
        
        if outputs['aux_logit'] is not None and self._cfg["use_aux_cls"] and self._task > 0:
            aux_targets = targets.clone()
            if self._cfg["aux_n+1"]:
                aux_targets[old_classes] = 0
                aux_targets[new_classes] -= num_old_classes - 1
            aux_loss = F.cross_entropy(outputs['aux_logit'], aux_targets)
            loss += aux_loss
            self._metrics["Aux loss"] += aux_loss.item()
            
        if self._old_model is not None and self._dist_loss['beta'] > 0:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)
                
            dist_loss = nn.KLDivLoss()(F.log_softmax(outputs['logit'][:,:num_old_classes]/self._dist_loss['T'], dim=1), \
                F.softmax(old_outputs['logit'].detach()/self._dist_loss['T'], dim=1)) * \
                self._dist_loss['T'] * self._dist_loss['T'] * self._dist_loss['beta'] * num_old_classes
            loss += dist_loss
            self._metrics["Dist loss"] += dist_loss.item()
                        
        if self._rank_loss['factor'] > 0 and self._task > 0:
            gt_index = torch.zeros(outputs['logit_bs'].size()).cuda()            
            gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
            gt_scores = outputs['logit_bs'].masked_select(gt_index)
            max_novel_scores = outputs['logit_bs'][:, num_old_classes:].topk(self._rank_loss['K'], dim=1)[0]
            hard_num = torch.nonzero(old_classes).size(0)
            if hard_num > 0:
                gt_scores = gt_scores[old_classes].view(-1, 1).repeat(1, self._rank_loss['K'])
                max_novel_scores = max_novel_scores[old_classes]
                assert(gt_scores.size() == max_novel_scores.size())
                assert(gt_scores.size(0) == hard_num)
                rank_loss = nn.MarginRankingLoss(margin=self._rank_loss['margin'])(gt_scores.view(-1, 1), \
                 max_novel_scores.view(-1, 1), torch.ones(hard_num*self._rank_loss['K']).cuda()) * self._rank_loss['factor']
            else:
                rank_loss = torch.tensor(0).cuda() #torch.zeros(1).cuda()
            loss += rank_loss
            self._metrics["Rank loss"] += rank_loss.item()
            
        return loss

    def _after_task(self, taski, inc_dataset):
        network = deepcopy(self._parallel_network)
        network.eval()
        if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
            self._ex.logger.info("save model")
            save_path = os.path.join(os.getcwd(), "ckpts", self._cfg["exp"]["name"])
            torch.save(network.cpu().state_dict(), "{}/step{}.ckpt".format(save_path, self._task))

        if (self._cfg["decouple"]['enable'] and taski > 0):
            if self._cfg["decouple"]["fullset"]:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
            else:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                       inc_dataset.targets_inc,
                                                       mode="balanced_train")

            # finetuning
            self._parallel_network.module.classifier.reset_parameters()
            finetune_last_layer(self._ex.logger,
                                self._parallel_network,
                                train_loader,
                                self._n_classes,
                                nepoch=self._decouple["epochs"],
                                lr=self._decouple["lr"],
                                scheduling=self._decouple["scheduling"],
                                lr_decay=self._decouple["lr_decay"],
                                weight_decay=self._decouple["weight_decay"],
                                loss_type="ce",
                                temperature=self._decouple["temperature"])
            network = deepcopy(self._parallel_network)
            if self._cfg["save_ckpt"]:
                save_path = os.path.join(os.getcwd(), "ckpts", self._cfg["exp"]["name"])
                torch.save(network.cpu().state_dict(), "{}/decouple_step{}.ckpt".format(save_path, self._task))

        if self._cfg["postprocessor"]["enable"]:
            self._update_postprocessor(inc_dataset)

        if self._cfg["infer_head"] == 'NCM':
            self._ex.logger.info("compute prototype")
            self.update_prototype()

        if self._memory_size.memsize != 0:
            self._ex.logger.info("build memory")
            self.build_exemplars(inc_dataset, self._coreset_strategy)

            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts", self._cfg['exp']['name'], "mem")
                memory = {
                    'x': inc_dataset.data_memory,
                    'y': inc_dataset.targets_memory,
                    'herding': self._herding_matrix
                }
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not (os.path.exists(f"{save_path}/mem_step{self._task}.ckpt") and self._cfg['load_mem']):
                    torch.save(memory, "{}/mem_step{}.ckpt".format(save_path, self._task))
                    self._ex.logger.info(f"Save step{self._task} memory!")

        self._parallel_network.eval()
        self._old_model = deepcopy(self._parallel_network)
        self._old_model.module.freeze()
        del self._inc_dataset.shared_data_inc
        self._inc_dataset.shared_data_inc = None

    def get_features(self, data_loader):
        preds, targets = [], []
        self._parallel_network.eval()
        with torch.no_grad():
            for i, (inputs, lbls) in enumerate(data_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                _preds = self._parallel_network(inputs)['feature']
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        return preds, targets
        
    def _eval_task(self, data_loader):
        if self._infer_head == "softmax":
            ypred, ytrue = self._compute_accuracy_by_netout(data_loader)
        elif self._infer_head == "NCM":
            ypred, ytrue = self._compute_accuracy_by_ncm(data_loader)
        else:
            raise ValueError()

        return ypred, ytrue

    def _compute_accuracy_by_netout(self, data_loader):
        preds, targets = [], []
        self._parallel_network.eval()
        with torch.no_grad():
            for i, (inputs, lbls) in enumerate(data_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                _preds = self._parallel_network(inputs)['logit']
                if self._cfg["postprocessor"]["enable"] and self._task > 0:
                    _preds = self._network.postprocessor.post_process(_preds, self._task_size)
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        return preds, targets

    def _compute_accuracy_by_ncm(self, loader):
        features, targets_ = extract_features(self._parallel_network, loader)
        targets = np.zeros((targets_.shape[0], self._n_classes), np.float32)
        targets[range(len(targets_)), targets_.astype("int32")] = 1.0

        class_means = (self._class_means.T / (np.linalg.norm(self._class_means.T, axis=0) + EPSILON)).T

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
        # Compute score for iCaRL
        sqd = cdist(class_means, features, "sqeuclidean")
        score_icarl = (-sqd).T
        return score_icarl[:, :self._n_classes], targets_

    def _update_postprocessor(self, inc_dataset):
        if self._cfg["postprocessor"]["type"].lower() == "bic":
            if self._cfg["postprocessor"]["disalign_resample"] is True:
                bic_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                     inc_dataset.targets_inc,
                                                     mode="train",
                                                     resample='disalign_resample')
            else:
                xdata, ydata = inc_dataset._select(inc_dataset.data_train,
                                                   inc_dataset.targets_train,
                                                   low_range=0,
                                                   high_range=self._n_classes)
                bic_loader = inc_dataset._get_loader(xdata, ydata, shuffle=True, mode='train')
            bic_loss = None
            self._network.postprocessor.reset(n_classes=self._n_classes)
            self._network.postprocessor.update(self._ex.logger,
                                               self._task_size,
                                               self._parallel_network,
                                               bic_loader,
                                               loss_criterion=bic_loss)
        elif self._cfg["postprocessor"]["type"].lower() == "wa":
            self._ex.logger.info("Post processor wa update !")
            self._network.postprocessor.update(self._network.classifier, self._task_size)

    def update_prototype(self):
        if hasattr(self._inc_dataset, 'shared_data_inc'):
            shared_data_inc = self._inc_dataset.shared_data_inc
        else:
            shared_data_inc = None
        self._class_means = update_classes_mean(self._parallel_network,
                                                self._inc_dataset,
                                                self._n_classes,
                                                self._task_size,
                                                share_memory=self._inc_dataset.shared_data_inc,
                                                metric='None')

    def build_exemplars(self, inc_dataset, coreset_strategy):
        save_path = os.path.join(os.getcwd(), f"ckpts/{self._cfg['exp']['name']}/mem/mem_step{self._task}.ckpt")
        if self._cfg["load_mem"] and os.path.exists(save_path):
            memory_states = torch.load(save_path)
            self._inc_dataset.data_memory = memory_states['x']
            self._inc_dataset.targets_memory = memory_states['y']
            self._herding_matrix = memory_states['herding']
            self._ex.logger.info(f"Load saved step{self._task} memory!")
            return

        if coreset_strategy == "random":
            from inclearn.tools.memory import random_selection

            self._inc_dataset.data_memory, self._inc_dataset.targets_memory = random_selection(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._ex.logger,
                inc_dataset,
                self._memory_per_class,
            )
        elif coreset_strategy == "iCaRL":
            from inclearn.tools.memory import herding
            data_inc = self._inc_dataset.shared_data_inc if self._inc_dataset.shared_data_inc is not None else self._inc_dataset.data_inc
            self._inc_dataset.data_memory, self._inc_dataset.targets_memory, self._herding_matrix = herding(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._herding_matrix,
                inc_dataset,
                data_inc,
                self._memory_per_class,
                self._ex.logger,
            )
        else:
            raise ValueError()

    def validate(self, data_loader):
        if self._infer_head == 'NCM':
            self.update_prototype()
        ypred, ytrue = self._eval_task(data_loader)
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        self._ex.logger.info(f"test top1acc:{test_acc_stats['top1']}")
