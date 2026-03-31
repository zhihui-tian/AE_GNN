#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn
#BSUB -G cmi
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

module load cuda/11.1 gcc/8
PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python
DAT=/g/g90/zhou6/lassen-space/data/Nd2Fe14B-before-100micron/512x512

$PYTHON train.py --outdir=experiment/Nd2Fe14B-before-100micron --data=$DAT \
  --resume=experiment/Nd2Fe14B-before-100micron/00003-stylegan3-r-512x512-gpus4-batch32-gamma8/network-snapshot-000480.pkl \
  --cfg=stylegan3-r --gpus=4 --batch=32 --gamma=8 --snap=15
