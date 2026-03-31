#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G amsdnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python
#DAT=$HOME/amsdnn/2D_forest/dataset5; stress=""
# DAT=$HOME/amsdnn/2D_forest/dataset6; stress="--var_stress=1e7"
DAT=$HOME/amsdnn/2D_forest/dataset7; stress="--var_stress=1e8"

device=0
for node_y in f+v; do # velocity force
if [[ "$node_y" == "force" ]]; then nfeat_in=10; else nfeat_in=10; fi
for stresslr in 1; do # 0 1
if [[ "$stresslr" == "1" ]]; then nfeat_edge_in=1; else nfeat_edge_in=0; fi
for nhid in 96 160  ; do
    for batch in 8 ; do
    for nmp in 2 3; do
    for act in gelu; do # silu or gelu
    for nlayer_mlp_encdec in  2; do
    for nlayer_mlp in 2 ; do
        for optimizer in adamw; do
        for wd in 5e-3 ; do
        for lr in 6e-4 ; do
        for dropout in 0.1 0; do
            for loss in L2 ; do
            DIR=experiment/forest2d7_force-batch${batch}_lr${lr}_loss${loss}_nmp${nmp}_nhid${nhid}_y${node_y}_opt${optimizer}_wd${wd}_stresslr${stresslr}_nlmlp${nlayer_mlp}_nlencdec${nlayer_mlp_encdec}_drop${dropout}_act${act}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --datatype=ddd_line_obstacle --single_dataset=1 --dataloader=geometric --nfeat_out=4  \
 --tskip=10 --dist_scale=1000 --normalize=1 --node_y=$node_y \
 --nfeat_in=$nfeat_in --nfeat_onehot=0 --nfeat_edge_in=$nfeat_edge_in --stress_lrange=$stresslr \
 --dim=2 --periodic $stress \
 --model=MeshGraphNets --gnnmodel=GNN --trainer=trainer_non_seq --act=$act --nfeat_hid=$nhid \
 --n_mpassing=$nmp --nlayer_mlp=$nlayer_mlp --nlayer_mlp_encdec=$nlayer_mlp_encdec \
 --batch=$batch --lr=$lr --nepoch=300 --epoch_size=20000 --loss=$loss --optimizer=$optimizer --wd=$wd \
 --dropout=$dropout --print_model \
 --print_freq=5000 --valid_freq=1 --scheduler=plateau --lr_decay_patience=10  \
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
done
done
done
done
wait
