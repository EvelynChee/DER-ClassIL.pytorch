import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F

from inclearn.tools import factory
from inclearn.convnet.imbalance import BiC, WA
from inclearn.convnet.classifier import CosineClassifier


class BasicNet(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
#             features = [convnet(x) for convnet in self.convnets]
#             features = torch.cat(features, 1)
            
            features = []
            prev_out = []
            for i, convnet in enumerate(self.convnets):
                cur_out = convnet(x, prev_out)
                for j in range(len(cur_out)-1):
                    if i == 0:
                        prev_out.append([cur_out[j]])
                    else:
                        prev_out[j].append(cur_out[j])
                features.append(cur_out[-1])
            features = torch.cat(features, dim=1)
#             features = cur_out[-1]
            
        else:
            features = self.convnet(x)

#         if self.ntask == 1:
        logits = self.classifier(features)
        if self.weight_normalization:
            logits, logits_bs = logits
        else:
            logits_bs = None
            
#         else:
#             logits1 = self.classifier(features)
#             logits2 = self.aux_classifier(features)
            
#             if self.weight_normalization:
#                 logits = torch.cat([logits1[0], logits2[0][:,1:]], dim=1)
#                 logits_bs = torch.cat([logits1[1], logits2[0][:,1:]], dim=1) 
#             else:
#                 logits = torch.cat([logits1, logits2[:,1:]], dim=1)
#                 logits_bs = None
        
#         if self.weight_normalization:
#             logits, logits_bs = logits
#         else:
#             logits_bs = None

        if features.shape[1] > self.out_dim: 
#             old_logits = torch.max(logits[:,:self.n_classes_old], dim=1, keepdim=True)
#             aux_logits = torch.cat((old_logits[0], logits[:,self.n_classes_old:]), dim=1)
#             if self.weight_normalization:
#                 aux_logits_bs = torch.cat((logits_bs[torch.arange(x.size(0)), old_logits[1].view(-1)].view(-1,1), logits_bs[:,self.n_classes_old:]), dim=1)
                      
            aux_logits = self.aux_classifier(features[:, -self.out_dim:])  
#             aux_logits = self.aux_classifier(features)  
            if self.weight_normalization:
                aux_logits, aux_logits_bs = aux_logits
            else:
                aux_logits_bs = None            
        else:
            aux_logits, aux_logits_bs = None, None
        return {'feature': features, 'logit_bs': logits_bs, 'logit': logits, 'aux_logit_bs':aux_logits_bs, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes_old = self.n_classes
        self.n_classes += n_classes
                                    
    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          prev_net=len(self.convnets),
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            
            ref_dict = self.convnets[-1].state_dict()
            tg_dict = new_clf.state_dict()            
            for k, v in ref_dict.items():
                if k in tg_dict and v.size() == tg_dict[k].size():
                    tg_dict[k] = v
            new_clf.load_state_dict(tg_dict)
                        
#             new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

#         if self.ntask == 1:
        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)
#         elif self.ntask == 2:
#             fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes)
#             fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets)-1)] = copy.deepcopy(self.classifier.weight.data)
#             fc.weight.data[:self.n_classes, -self.out_dim:] = 0.0
#         else:
#             fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes)
#             fc.weight.data[:-n_classes, :-self.out_dim] = copy.deepcopy(self.classifier.weight.data)
#             fc.weight.data[-n_classes:, :-self.out_dim] = copy.deepcopy(self.aux_classifier.weight.data[1:])
#             fc.weight.data[:, -self.out_dim:] = 0.0
            
        
        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
#             fc.weight.data[:self.n_classes, self.out_dim * (len(self.convnets) - 1):] = 0.0
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier
