#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G mesosim
# BSUB -q pbatch
# BSUB -W 12:00

##BSUB -q pdebug
##BSUB -W 00:30

#BSUB -nnodes 1



module load cuda/12.2.2
module load cmake/3.29.2
module load gcc/12.2.1
module list

source activate /usr/WS2/tian9/myenv_backup
PYTHON=/usr/WS2/tian9/myenv_backup/bin/python


# DAT=ls
DAT=/usr/WS1/amsdnn/data/grain2d

device=0
for n_in in 1; do
for n_out in 1 ; do
for nhid in 96; do
    for batch in 4; do   ### batch size has influence
    for nmp in 3; do
    # for nmp in 8 16; do
    for model in NPS; do #NPS_autoencoder
    # for model in NPS; do #NPS
    # for model in NPS; do
        for noise in 1e-3; do
        # for lr in 4e-3; do
        for lr in 1e-4; do
            for loss in L2 ; do
            # for nae in 1 2; do
            # for nencdec in 1 2; do
            for nae in 1; do
            for nencdec in 1; do
            ae_stride=`python -c "print(','.join(map(str,[1,2,2,2][:$nae])))"`
            ae_block=`python -c "print(','.join(map(str,[2,2,2,2][:$nae])))"`
            # nfeat_autoencoder=`python -c "print(4**$nae)"` # --nfeat_autoencoder=$nfeat_autoencoder
            DIR=experiment/grain_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_nae${nae}_nencdec${nencdec}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=64 --nfeat_in=1 --periodic --pointgroup=4m \
 --dim=2 --periodic \
 --model=MeshGraphNets --gnnmodel=$model --autoencoder=rev2wae --feat_out_method=id --trainer=MeshGraphNets --act=relu --nfeat_hid=$nhid \
 --n_mpassing=$nmp --nstrides_2wae=$ae_stride --nblocks_2wae=$ae_block --nlayer_mlp_encdec=$nencdec \
 --optimizer=adamw \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --noise_op="add_normal/0:1/$noise" --nepoch=50 --epoch_size=2000 --n_out_valid=5 --loss=$loss \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=8 --n_traj_out=10 \
 --n_out_predict=100 --clip_step_valid=10 --mode=predict --log_mem &>>$DIR/log &
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
