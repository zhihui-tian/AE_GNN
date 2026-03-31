#!/bin/env bash

DIR=$1
#DIR=experiment/mgn_sigmoid_amr-forest2d5-batch32_lr3e-4_nin1_nout1_noise1e-4_RNN0_lossL1_wt6_nmp13_nhid64_drop1e-2/;  

cmd=`grep mode=train $DIR/config.txt |tail -n1 |sed 's/mode=train/mode=eval/;s/clip_step_valid=30/clip_step_valid=3000/;s/n_out_valid=5 /n_out_valid=500 /;s/n_traj_out=10 /n_traj_out=30 /' `
$cmd
calculator_array.py "np.concatenate(( $DIR/gt.npy , $DIR/pd.npy [...,:1]),-1)" -o $DIR/combined.npy
animate-2d.py --rv --tskip=10 -i=0:3 $DIR/comb*.npy
