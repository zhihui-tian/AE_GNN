#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G msgnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python
DAT=$HOME/amsdnn/data/CHE2d
dat=CHE2d

device=0
for ker in 3; do
for n_in in 3; do
for n_out in 3 ; do
for nhid in 64 ; do
    for batch in 8; do
    for nmp in 3 4 5 6 ; do
    for model in convlstm; do #NPS_autoencoder
        for noise in 1e-5; do
        for lr in 0.8e-4 ; do
        for RNN in 1; do
            for loss in L2 ; do
            DIR=experiment/${dat}_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_ker${ker}_RNN${RNN}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=64 --nfeat_in=1 --periodic --pointgroup=4m \
 --dim=2 --periodic --channel_first \
 --model=PhyDNet.$model  --act=silu --nfeat_hid=$nhid --n_mpassing=$nmp --kernel_size=$ker --RNN=$RNN \
 --optimizer=adamw \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --noise_op="add_normal/0:1/$noise" --nepoch=300 --epoch_size=2500 --n_out_valid=5 --loss=$loss \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=8 --n_traj_out=10 \
 --n_out_predict=100 --clip_step_valid=10 --mode=train &>>$DIR/log &
            device=$(((device+1)%4))
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
