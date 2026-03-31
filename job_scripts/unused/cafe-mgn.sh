#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G amsdnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python

########## choose which dataset and associated parameters
DAT=$HOME/data/cafe_gnn/ALE3DCAFE_PANDAS_iso_40
frame_shape=125,125,78
dat_name=cafe_iso
selected_chop=slice

# DAT=$HOME/data/cafe_gnn/MINI/ALE3DCAFE_PANDAS_iso_40
# frame_shape=31,31,78
# dat_name=cafeMINI_iso
# selected_chop=none
############################################################

device=0
for chop in $selected_chop; do # amr or slice or none
    if [[ $chop == 'amr' ]]; then
        chop_method='--amr=pointwise_v2'
    elif [[ $chop == 'slice' ]]; then
        chop_method='--amr= --data_aug=slice --slice_op=-1,-1,10'
    else
        chop_method='--amr= --data_aug= '
    fi
for n_in in 1; do
for n_out in 1 ; do
for wd in 1e-3; do
for nhid in 32 48; do
    for batch in 1 2; do
    for nmp in 1 ; do
        for optimizer in adamw; do
        for noise in 1e-5; do #1e-3
        for lr in 5e-4; do # 5e-4 ; do
            for RNN in 0; do #150 180 80 128
            for loss in L2; do #L2 L1_wt2 L1_wt10
            DIR=experiment/${dat_name}_gnn-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_RNN${RNN}_loss${loss}_nmp${nmp}_nhid${nhid}_opt${optimizer}_wd${wd}_chop${chop}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=$frame_shape --data_slice=':,:400,...,[0,3,4,5,1,2]' \
 --dim=3 \
 --model=MeshGraphNets.cafe_gnn_model --trainer=MeshGraphNets.cafe_gnn_model --act=silu --nfeat_hid=$nhid  `#--amr=pointwise` \
 --n_mpassing=$nmp --RNN=$RNN \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --nepoch=300 --epoch_size=2000 --n_out_valid=1 --loss=$loss --optimizer=$optimizer --wd=$wd \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=8 --n_traj_out=10 --amr_buffer=1 \
 $chop_method \
 --n_out_predict=100 --clip_step_valid=1 --mode=train &>>$DIR/log &
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
wait
