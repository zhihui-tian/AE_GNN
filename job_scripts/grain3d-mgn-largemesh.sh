#!/usr/bin/env bash
source /usr/WS2/tian9/tuolu_venv2/bin/activate
PYTHON=/usr/WS2/tian9/tuolu_venv2/bin/python

device=0
for dimension in 128; do
# dimension=160
infer_mode=original
# infer_mode=optimize




# DAT=/usr/WS2/tian9/KMC_3D_2_4_bh_slow ###(32,32,32)  ### for figure 2

# DAT=/usr/WS2/tian9/KMC_3D_2_5_bold_hot ###(159,20,64,64,64)  --n_out_valid=19

# DAT=/usr/WS2/tian9/KMC_3D_predict ###(96,96,96) (1,75,96,96,96) or (96,75,96,96,96)  --n_out_valid=74  ### for 3d visualization
DAT=/usr/WS2/tian9/fake3d_data/KMC_3D_predict_$dimension
# DAT=/usr/WS2/tian9/fake3d_data/KMC_3D_predict_x5  #480 out of memory

# DAT=/usr/WS2/tian9/KMC_3D_stats_long_slow ###(96,96,96) (6,300,96,96,96)


for n_in in 1; do
for n_out in 1 ; do
for nhid in 96; do
    for batch in 4; do   ### batch size has influence
    for nmp in 3; do

    for model in NPS_autoencoder; do #NPS_autoencoder
    # for model in NPS; do

        for noise in 1e-3; do
        for lr in 1e-4; do
            for loss in L2 ; do

            for nae in 1 2 3; do
            # for nae in 1; do

            for nencdec in 1; do
            ae_stride=`python -c "print(','.join(map(str,[1,2,2,2][:$nae])))"`
            ae_block=`python -c "print(','.join(map(str,[2,2,2,2][:$nae])))"`
            # nfeat_autoencoder=`python -c "print(4**$nae)"` # --nfeat_autoencoder=$nfeat_autoencoder
            DIR=experiment/grain_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_nae${nae}_nencdec${nencdec}
            mkdir -p $DIR
            echo $DIR $device

            if [ "$model" = "NPS" ]; then
                LOG_FILE=$DIR/log_gnn_${dimension}
            elif [[ "$model" == "NPS_autoencoder" && "$infer_mode" == "original" ]]; then
                LOG_FILE=$DIR/log_org_${dimension}
            else
                LOG_FILE=$DIR/log_lat_${dimension}
            fi

            echo $model $infer_mode
            echo $LOG_FILE

            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=$dimension --nfeat_in=1 --periodic --pointgroup=1 \
 --dim=3 --periodic \
 --model=MeshGraphNets --gnnmodel=$model --autoencoder=rev2wae --feat_out_method=id --trainer=MeshGraphNets --act=relu --nfeat_hid=$nhid \
 --n_mpassing=$nmp --nstrides_2wae=$ae_stride --nblocks_2wae=$ae_block --nlayer_mlp_encdec=$nencdec \
 --optimizer=adamw \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --noise_op="add_normal/0:1/$noise" --nepoch=5 --epoch_size=2000 --n_out_valid=24 --infer_mode=$infer_mode --loss=$loss \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=8 --n_traj_out=100 \
 --n_out_predict=100 --clip_step_valid=1 --mode=train --log_mem &>>$LOG_FILE &
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
