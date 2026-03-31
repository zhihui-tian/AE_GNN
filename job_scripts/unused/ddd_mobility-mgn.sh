#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G msgnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python
# DAT=$HOME/amsdnn/3D_bulk/dataset1;
DAT=$HOME/amsdnn/3D_bulk/dataset2

device=0
for node_y in vel_inst; do # velocity force
if [[ "$node_y" == "force" ]]; then nfeat_in=9; else nfeat_in=12; fi
for cut in 0; do
for nhid in 160 ; do
    for batch in 8; do
    for nmp in 4; do
        for optimizer in adamw; do
        for wd in 4e-2 6e-2; do
        for lr in 4e-4 6e-4; do
            for loss in L2 ; do
            DIR=experiment/ddd3d_force-batch${batch}_lr${lr}_loss${loss}_nmp${nmp}_nhid${nhid}_y${node_y}_opt${optimizer}_wd${wd}_cut${cut}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --datatype=ddd_line3d --single_dataset=1 --dataloader=geometric --nfeat_in=3 --nfeat_out=3 \
 --tskip=5 --dist_scale=1 --force_scale=1e9 --normalize=1 --node_y=$node_y --nonlink_cutoff=$cut \
 --dim=3 --periodic \
 --model=MeshGraphNets --gnnmodel=GNN --trainer=trainer_non_seq --act=silu --nfeat_in=$nfeat_in --nfeat_onehot=0 --nfeat_edge_in=3 --nfeat_hid=$nhid \
 --n_mpassing=$nmp \
 --batch=$batch --lr=$lr --nepoch=300 --epoch_size=2000 --loss=$loss --optimizer=$optimizer --wd=$wd \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=10 \
 --mode=train &>>$DIR/log &
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
