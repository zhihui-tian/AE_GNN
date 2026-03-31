#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G amsdnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python
DAT=$HOME/data/cafe_gnn/MINI/ALE3DCAFE_PANDAS_iso_40
frame_shape=31,31,78

device=0
for n_in in 3; do
for n_out in 3 ; do
for ngram in 1; do
    for batch in 2; do
    for nmp in 2 ; do
        for noise in 1e-5 ; do #1e-3
        for lr in 3e-4; do #3e-3 8e-3 1e-3 3e-3 6e-4 3e-4; do
            for RNN in 1 0; do #150 180 80 128
            for loss in L2 L1 ; do #L2 L1_wt2 L1_wt10
# for act in mish; do #mish swish relu tanh
            DIR=experiment/cafe_iso_phydnet-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_RNN${RNN}_loss${loss}_nmp${nmp}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR --n_GPUs=1 \
 --data=$DAT --dataset=longclip --frame_shape=$frame_shape --nfeat_in=1 --data_slice='...,0:1:2' \
 --dim=3 \
 --model=PhyDNet --trainer=PhyDNet --channel_first --nfeat_hid=64 --encdec_unet=1 --dx=0 --last_act= \
 --n_mpassing=$nmp --RNN=$RNN \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --noise=$noise --nepoch=3000 --epoch_size=1000 --n_out_valid=5 --loss=$loss --ngram=$ngram \
 --data_aug=slice --slice_op=28,28,28  --pointgroup=1 \
 --print_freq=250 --valid_freq=1 --scheduler=plateau --lr_decay_patience=8 --n_traj_out=10 \
 --n_out_predict=10 --clip_step_valid=30 --mode=train &>>$DIR/log &
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
wait

# DIR=experiment/phydnet-grain-batch24_lr5e-4_nin1_nout1_noise2e-2_RNN0_lastnone
# python -m  PhyDNet.main  --data=/g/g90/zhou6/data/grain/ --dir=$DIR --dataset=longclip --dim=2 --periodic --RNN=0 --last_act= --mode=predict --batch=100 --lr=5e-4 --n_in=1 --n_out=0 --noise=2e-2 --n_rollout=15 --n_predict=1 --file_test=tmp.npy --clip_step=1
# animate-2d.py $DIR/predict.npy
