#!/usr/bin/env bash
# name='Inference'
debug='0'
comments='None'
trial=1
devices=2
pretrain=50
step=10
name="cifar100_b${pretrain}_s${step}_trial${trial}_ours"


if [ ${debug} -eq '0' ]; then
    CUDA_VISIBLE_DEVICES="${devices}" python3 -m main visualize with "./configs/b${pretrain}_s${step}_ours.yaml" \
        exp.name="${name}" \
        exp.savedir="./logs/" \
        exp.ckptdir="./logs/" \
        exp.tensorboard_dir="./tensorboard/" \
        trial=${trial} \
        save_ckpt=False \
        save_mem=False \
        --name="${name}" \
        -D \
        -p \
        -c "${comments}" \
        --force \
        # --mongo_db=10.10.10.100:30620:classil
else
    CUDA_VISIBLE_DEVICES="${devices}" python3 -m main visualize with "./configs/b${pretrain}_s${step}_ours.yaml" \
        exp.name="${name}_debug" \
        exp.savedir="./logs/" \
        exp.ckptdir="./logs/" \
        exp.tensorboard_dir="./tensorboard/" \
        exp.debug=True \
        trial=${trial} \
        save_ckpt=False \
        save_mem=False \
        load_mem=True \
        --name="${name}" \
        -D \
        -p \
        --force \
        #--mongo_db=10.10.10.100:30620:debug
fi
