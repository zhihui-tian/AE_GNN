#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G mesosim
##BSUB -q pbatch
#BSUB -q pdebug
#BSUB -nnodes 1
##BSUB -W 12:00
#BSUB -W 00:30

module load cuda/12.2.2
module load cmake/3.29.2
module load gcc/12.2.1
module list

source activate /g/g92/tian9/myenv

# PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python
PYTHON=$HOME/myenv/bin/python

# DAT=$HOME/amsdnn/data/grain2d/
DAT=/usr/WS1/amsdnn/data/grain2d
device=0
for ker in 5; do
for n_in in 1; do
for n_out in 1 ; do
for nhid in 96 ; do
    for batch in 16; do
    # for nmp in 6 8 9 10 ; do
    for nmp in 6 ; do
    for model in  convnext; do
        for noise in 1e-3 ; do
        for lr in 1e-4; do
        for RNN in 0; do
            for loss in L2 ; do
            DIR=experiment/grain2d_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_ker${ker}_RNN${RNN}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=64 --nfeat_in=1 --periodic --pointgroup=4m \
 --dim=2 --periodic --channel_first \
 --model=NPS.model.$model  --act=silu --nfeat_hid=$nhid --n_mpassing=$nmp --kernel_size=$ker --RNN=$RNN \
 --optimizer=adamw \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --noise_op="add_normal/0:1/$noise" --nepoch=2 --epoch_size=2000 --n_out_valid=5 --loss=$loss \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=8 --n_traj_out=10 \
 --n_out_predict=100 --clip_step_valid=10 --mode=train &>>$DIR/log &
            device=$(((device+1)%2))
done
done
done
done
done
done
done
done
done
done
done
wait
