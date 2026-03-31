#!/usr/bin/env bash
source /usr/WS2/tian9/tuolu_venv2/bin/activate
PYTHON=/usr/WS2/tian9/tuolu_venv2/bin/python

device=0
for dimension in 96; do
# for dimension in 96; do
# dimension=160
infer_mode=original
# infer_mode=optimize




DAT=/usr/WS2/tian9/KMC_3D_2_4_bh_slow ###(25,32,32,32)  ### for figure 2 #--n_out_valid=24
# DAT=/usr/WS2/tian9/KMC_3D_stats_long_slow  #(6,300,96,96,96), match the 32^3 training dataset speed
# DAT=/usr/WS2/tian9/KMC_3D_slow_compress_corr ###clone of KMC_3D_stats_long_slow (6, 225, 96, 96, 96, 1) for train and (6, 75, 96, 96, 96, 1) for valid
# (DAT=/usr/WS2/tian9/KMC_3D_predict ###(96,96,96) (1,75,96,96,96) or (96,75,96,96,96)  --n_out_valid=74  ### for 3d visualization, do not use this one)


################################################################### AE+GNN
for n_in in 1; do
for n_out in 1 3 5 7 ; do
# for n_out in 5 ; do
for nhid in 96; do
    for batch in 4; do   ### batch size has influence
    # for nmp in 3; do
    for nmp in 3; do

    for model in NPS_autoencoder; do #NPS_autoencoder
    # for model in NPS; do

        for noise in 1e-3; do
        for lr in 1e-4; do
            for loss in L2 ; do

            # for nae in 1 2 3; do
            for nae in 1; do


            for nencdec in 1; do
            ae_stride=`python -c "print(','.join(map(str,[1,2,2,2][:$nae])))"`
            ae_block=`python -c "print(','.join(map(str,[2,2,2,2][:$nae])))"`
            # nfeat_autoencoder=`python -c "print(4**$nae)"` # --nfeat_autoencoder=$nfeat_autoencoder

            DIR=experiment/60epoch/grain_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_nae${nae}_nencdec${nencdec}
            # DIR=experiment/silu_l2/grain_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_nae${nae}_nencdec${nencdec}

            mkdir -p $DIR
            echo $DIR $device

            if [ "$model" = "NPS" ]; then
                LOG_FILE=$DIR/log_gnn_${dimension}_infer
            elif [[ "$model" == "NPS_autoencoder" && "$infer_mode" == "original" ]]; then
                LOG_FILE=$DIR/log_org_${dimension}_infer
            else
                LOG_FILE=$DIR/log_lat_${dimension}_infer
            fi

            echo $model $infer_mode
            echo $LOG_FILE

### high compression check
# for n_in in 1; do
# for n_out in 1 5; do
# for nhid in 96; do
#     for batch in 4; do   ### batch size has influence
#     for nmp in 3; do

#     for model in NPS_autoencoder; do #NPS_autoencoder
#     # for model in NPS; do

#         for noise in 1e-3; do
#         for lr in 1e-4; do
#             for loss in L2 ; do

#             for nae in 1 2 3; do
#             # for nae in 3; do
#             for nencdec in 1; do
#             ae_stride=`python -c "print(','.join(map(str,[1,2,2,2][:$nae])))"`
#             ae_block=`python -c "print(','.join(map(str,[2,2,2,2][:$nae])))"`
#             # nfeat_autoencoder=`python -c "print(4**$nae)"` # --nfeat_autoencoder=$nfeat_autoencoder

#             DIR=experiment/high_compression_slow_splitdata/slow5_step/grain_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_nae${nae}_nencdec${nencdec}
#             # DIR=experiment/silu_l2/grain_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_nae${nae}_nencdec${nencdec}

#             mkdir -p $DIR
#             echo $DIR $device

#             if [ "$model" = "NPS" ]; then
#                 LOG_FILE=$DIR/log_gnn_${dimension}_infer
#             elif [[ "$model" == "NPS_autoencoder" && "$infer_mode" == "original" ]]; then
#                 LOG_FILE=$DIR/log_org_${dimension}_infer
#             else
#                 LOG_FILE=$DIR/log_lat_${dimension}_infer
#             fi

#             echo $model $infer_mode
#             echo $LOG_FILE

################################################################### GNN only
# for n_in in 1; do
# # for n_out in 1 3 5 7 ; do
# for n_out in 1 ; do
# for nhid in 96; do
#     for batch in 4; do   ### batch size has influence
#     # for nmp in 3; do
#     for nmp in 3; do

#     for model in NPS; do #NPS_autoencoder
#     # for model in NPS; do

#         for noise in 1e-3; do
#         for lr in 1e-4; do
#             for loss in L2 ; do

#             # for nae in 1 2 3; do
#             for nae in 1; do


#             for nencdec in 1; do
#             ae_stride=`python -c "print(','.join(map(str,[1,2,2,2][:$nae])))"`
#             ae_block=`python -c "print(','.join(map(str,[2,2,2,2][:$nae])))"`
#             # nfeat_autoencoder=`python -c "print(4**$nae)"` # --nfeat_autoencoder=$nfeat_autoencoder

#             DIR=experiment/GNN_only_relu/grain_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_nae${nae}_nencdec${nencdec}
#             # DIR=/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS-main/experiment/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1

#             mkdir -p $DIR
#             echo $DIR $device

#             if [ "$model" = "NPS" ]; then
#                 LOG_FILE=$DIR/log_gnn_${dimension}_infer
#             elif [[ "$model" == "NPS_autoencoder" && "$infer_mode" == "original" ]]; then
#                 LOG_FILE=$DIR/log_org_${dimension}_infer
#             else
#                 LOG_FILE=$DIR/log_lat_${dimension}_infer
#             fi

#             echo $model $infer_mode
#             echo $LOG_FILE




            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=$dimension --nfeat_in=1 --periodic --pointgroup=1 \
 --dim=3 --periodic \
 --model=MeshGraphNets --gnnmodel=$model --autoencoder=rev2wae --feat_out_method=id --trainer=MeshGraphNets --act=relu --nfeat_hid=$nhid \
 --n_mpassing=$nmp --nstrides_2wae=$ae_stride --nblocks_2wae=$ae_block --nlayer_mlp_encdec=$nencdec \
 --optimizer=adamw \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --noise_op="add_normal/0:1/$noise" --nepoch=60 --epoch_size=2000 --n_out_valid=224 --infer_mode=$infer_mode --loss=$loss \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=8 --n_traj_out=100 \
 --n_out_predict=100 --clip_step_valid=1 --mode=predict --resume=-2 --log_mem &>>$LOG_FILE &
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


# n_traj_out 控制rollout次数
# n_out_valid 控制每个rollout多长

# clip_step_valid 大概率没用, 没走trainer.py中 if 分支
# n_out_predict 大概率没用，没走trainer.py中if分支













# for n_in in 1; do
# for n_out in 5 ; do
# for nhid in 96; do
#     for batch in 8; do   ### batch size has influence
#     for nmp in 16; do

#     for model in NPS_autoencoder; do #NPS_autoencoder
#     # for model in NPS; do

#         for noise in 1e-3; do
#         for lr in 1e-4; do
#             for loss in L2 ; do

#             for nae in 2; do
#             # for nae in 1; do
#             for nencdec in 2; do
#             ae_stride=`python -c "print(','.join(map(str,[1,2,2,2][:$nae])))"`
#             ae_block=`python -c "print(','.join(map(str,[2,2,2,2][:$nae])))"`
#             # nfeat_autoencoder=`python -c "print(4**$nae)"` # --nfeat_autoencoder=$nfeat_autoencoder

#             DIR=/usr/WS2/tian9/ethanresult/NPS-runs/KMC_3D_2_4_bh_slow_grain2d_NPS_autoencoder/batch8_lr1e-3_nin1_nout5-noiseadd_normal5e-1_lossL1_nmp16_nae2_nencdec2_dropout0
            
#             mkdir -p $DIR
#             echo $DIR $device

#             if [ "$model" = "NPS" ]; then
#                 LOG_FILE=$DIR/log_gnn_${dimension}
#             elif [[ "$model" == "NPS_autoencoder" && "$infer_mode" == "original" ]]; then
#                 LOG_FILE=$DIR/log_org_${dimension}
#             else
#                 LOG_FILE=$DIR/log_lat_${dimension}
#             fi

#             echo $model $infer_mode
#             echo $LOG_FILE