#!/usr/bin/env bash
debug='0'
comments='None'
trial=1
devices=3
pretrain=0
step=10
name="cifar100_b${pretrain}_s${step}_trial${trial}_ours_v4"


if [ ${debug} -eq '0' ]; then
    CUDA_VISIBLE_DEVICES="${devices}" python3 -m main train with "./configs/b${pretrain}_s${step}_ours.yaml" \
        exp.name="${name}" \
        exp.savedir="./logs/" \
        exp.ckptdir="./logs/" \
        exp.tensorboard_dir="./tensorboard/" \
        trial=${trial} \
        --name="${name}" \
        -D \
        -p \
        -c "${comments}" \
        --force \
        # --mongo_db=10.10.10.100:30620:classil
else
    CUDA_VISIBLE_DEVICES="${devices}" python3 -m main train with "./configs/b${pretrain}_s${step}.yaml" \
        exp.name="${name}_debug" \
        exp.savedir="./logs/" \
        exp.ckptdir="./logs/" \
        exp.tensorboard_dir="./tensorboard/" \
        exp.debug=True \
        trial=${trial} \
        --name="${name}" \
        -D \
        -p \
        --force \
        #--mongo_db=10.10.10.100:30620:debug
fi
