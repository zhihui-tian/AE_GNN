#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G msgnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python
DAT=$HOME/amsdnn/grain/ 

device=0
for n_in in 1 ; do
for n_out in 1 ; do
    for batch in 100; do
    for ker in 7 ; do
        for noise in 1e-4 1e-2; do
        for lr in 4e-3 8e-3 2e-2 5e-2; do #1e-3 3e-3 6e-4 3e-4; do
            for RNN in 0; do #150 180 80 128
# for act in mish; do #mish swish relu tanh
            DIR=experiment/crevnet-grain-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_RNN${RNN}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m CrevNet.only_2d.train \
 --data=$DAT --model_dir=$DIR --dataset=longclip --frame_shape=64,64 --channels=1  \
 --dim=2 --periodic --g_dim=128 --rnn_size=128 --RNN=$RNN \
 --mode=train \
 --batch_size=$batch --lr=$lr --n_past=${n_in} --n_future=${n_out} --noise=$noise --niter=10000 --epoch_size=400 --n_out_test=5 \
 --n_rollout=10 --n_predict=100 --clip_step_test=10 &>>$DIR/log &
            device=$(((device+1)%4))
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
