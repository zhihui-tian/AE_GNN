#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G amsdnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python
DAT=$HOME/amsdnn/2D_forest/dataset5

device=0
for n_in in 1; do
for n_out in 1 ; do
for ngram in 1; do
for nhid in 64; do
    for batch in 32; do
    for nmp in 18 21 ; do
        for noise in 1e-4; do #1e-3
        for drop in 3e-1 3e-2; do #1e-3
        for lr in 3e-4; do #3e-3 8e-3 1e-3 3e-3 6e-4 3e-4; do
            for RNN in 0; do #150 180 80 128
            for loss in L1_wt6 ; do #L2 L1_wt2 L1_wt10
# for act in mish; do #mish swish relu tanh
            DIR=experiment/mgn_sigmoid_amr-forest2d5-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_RNN${RNN}_loss${loss}_nmp${nmp}_nhid${nhid}_drop${drop}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=64,256 --nfeat_in=2 --periodic --pointgroup=mx --data_slice='...,::2' --data_preprocess='np.clip(x,0,1)' \
 --dim=2 --periodic \
 --model=MeshGraphNets --gnnmodel=NPS --amr=pointwise --feat_out_method=sigmoid,fix --trainer=MeshGraphNets --act=silu --nfeat_hid=$nhid \
 --n_mpassing=$nmp --RNN=$RNN \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --noise_op="drop/0:1/$drop,add_normal/0:1/$noise" --nepoch=300 --epoch_size=2000 --n_out_valid=5 --loss=$loss --ngram=$ngram \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=6 --n_traj_out=10 \
 --n_out_predict=100 --clip_step_valid=30 --mode=train &>>$DIR/log &
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

# DIR=experiment/phydnet-grain-batch24_lr5e-4_nin1_nout1_noise2e-2_RNN0_lastnone
# python -m  PhyDNet.main  --data=/g/g90/zhou6/data/grain/ --dir=$DIR --dataset=longclip --dim=2 --periodic --RNN=0 --last_act= --mode=predict --batch=100 --lr=5e-4 --n_in=1 --n_out=0 --noise=2e-2 --n_rollout=15 --n_predict=1 --file_test=tmp.npy --clip_step=1
# animate-2d.py $DIR/predict.npy
