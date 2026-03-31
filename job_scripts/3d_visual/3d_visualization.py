

"""
======================================================
 Script Name:    2d_visualization.py
 Author:         Zhihui Tian
 Created:        2025-09-18
 Description:    
     This script used for visualize the 3d results for gt, inference in original space and inference in the latent space. used to generate the figure for 3d visualization in the paper
======================================================
"""


from utility_plots import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pyvista as pv


"""32"""

# gt = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')
# # pd_org = np.load('/usr/WS2/tian9/ethanresult/NPS-runs/KMC_3D_2_4_bh_slow_grain2d_NPS_autoencoder/batch8_lr1e-3_nin1_nout5-noiseadd_normal5e-1_lossL1_nmp16_nae2_nencdec2_dropout0/pd_32.npy')
# gnn_only = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS-main/experiment/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# step1 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/steps_tune/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# step3 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/steps_tune/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout3_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# step5 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/steps_tune/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# # step7 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/steps_tune/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout7_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# # print(np.array_equal(gt[0,0],pd_org[0,0]))

# plot_3_snapshots(gt[0,0,:,:,:,0],
#                  gt[0,5,:,:,:,0],
#                  gt[0,10,:,:,:,0],
#                  gt[0,19,:,:,:,0],cmap = 'Greys_r',save_path ='./gt.png')

# plot_3_snapshots(gnn_only[0,0,:,:,:,0],
#                  gnn_only[0,5,:,:,:,0],
#                  gnn_only[0,10,:,:,:,0],
#                  gnn_only[0,19,:,:,:,0],cmap = 'Greys_r',save_path ='./gnn.png')

# plot_3_snapshots(step1[0,0,:,:,:,0],
#                  step1[0,5,:,:,:,0],
#                  step1[0,10,:,:,:,0],
#                  step1[0,19,:,:,:,0],cmap = 'Greys_r',save_path ='./step1.png')

# plot_3_snapshots(step3[0,0,:,:,:,0],
#                  step3[0,5,:,:,:,0],
#                  step3[0,10,:,:,:,0],
#                  step3[0,19,:,:,:,0],cmap = 'Greys_r',save_path ='./step3.png')

# plot_3_snapshots(step5[0,0,:,:,:,0],
#                  step5[0,5,:,:,:,0],
#                  step5[0,10,:,:,:,0],
#                  step5[0,19,:,:,:,0],cmap = 'Greys_r',save_path ='./step5.png')

# # plot_3_snapshots(step7[0,0,:,:,:,0],
# #                  step7[0,5,:,:,:,0],
# #                  step7[0,10,:,:,:,0],
# #                  step7[0,19,:,:,:,0],cmap = 'Greys_r',save_path ='./step7.png')


# gt = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')
# gnn_only = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS-main/experiment/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd32.npy')
# gnn_only2=np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/GNN_only_relu/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')


# plot_3_snapshots(gt[0,0,:,:,:,0],
#                  gt[0,5,:,:,:,0],
#                  gt[0,10,:,:,:,0],
#                  gt[0,19,:,:,:,0],cmap = 'Greys_r',save_path ='./gt.png')

# plot_3_snapshots(gnn_only[0,0,:,:,:,0],
#                  gnn_only[0,5,:,:,:,0],
#                  gnn_only[0,10,:,:,:,0],
#                  gnn_only[0,19,:,:,:,0],cmap = 'Greys_r',save_path ='./gnn.png')

# plot_3_snapshots(gnn_only2[0,0,:,:,:,0],
#                  gnn_only2[0,5,:,:,:,0],
#                  gnn_only2[0,10,:,:,:,0],
#                  gnn_only2[0,19,:,:,:,0],cmap = 'Greys_r',save_path ='./gnnonly.png')



######################################################################################"""96 extrapolation"""
### slow 96^3
# gt = np.load('/usr/WS2/tian9/KMC_3D_stats_long_slow/valid.npy')
# # gnn_only=np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS-main/experiment/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd96.npy')
# gnn_only=np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/GNN_only_relu/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd96.npy')


# pd_org = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/60epoch/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd96_org_slow.npy')
# pd_lat10 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/60epoch/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd_latent1.npy')
# pd_lat100 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/60epoch/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd_latent2.npy')
# pd_lat200 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/60epoch/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd_latent3.npy')
# print(np.array_equal(gt[0,0],pd_org[0,0]))

# plot_3_snapshots(gt[0,0,:,:,:,0],
#                  gt[0,9,:,:,:,0],
#                  gt[0,99,:,:,:,0],
#                  gt[0,199,:,:,:,0],cmap = 'Greys_r',save_path ='./gt96new.png')

# plot_3_snapshots(gnn_only[0,0,:,:,:,0],
#                  gnn_only[0,9,:,:,:,0],
#                  gnn_only[0,99,:,:,:,0],
#                  gnn_only[0,199,:,:,:,0],cmap = 'Greys_r',save_path ='./gnn_only.png')

# plot_3_snapshots(pd_org[0,0,:,:,:,0],
#                  pd_org[0,9,:,:,:,0],
#                  pd_org[0,99,:,:,:,0],
#                  pd_org[0,199,:,:,:,0],cmap = 'Greys_r',save_path ='./pd96_org_slow.png')

# plot_3_snapshots(pd_lat10[0,0,:,:,:,0],
#                  pd_lat10[0,1,:,:,:,0],
#                  pd_lat100[0,1,:,:,:,0],
#                  pd_lat200[0,1,:,:,:,0],cmap = 'Greys_r',save_path ='./pd96_lat_slow.png')


#"""more compression"""
gt = np.load('/usr/WS2/tian9/KMC_3D_stats_long_slow/valid.npy')

# pd_org_compre2 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slowtrain_5step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# pd_org_compre4 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slowtrain_5step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd.npy')


# plot_3_snapshots(gt[0,0,:,:,:,0],
#                  gt[0,19,:,:,:,0],
#                  gt[0,49,:,:,:,0],
#                  gt[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./gt96new.png')

# plot_3_snapshots(pd_org_compre2[0,0,:,:,:,0],
#                  pd_org_compre2[0,19,:,:,:,0],
#                  pd_org_compre2[0,49,:,:,:,0],
#                  pd_org_compre2[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./pd96_org_slow_compression2_5step.png')


# plot_3_snapshots(pd_org_compre4[0,0,:,:,:,0],
#                  pd_org_compre4[0,19,:,:,:,0],
#                  pd_org_compre4[0,49,:,:,:,0],
#                  pd_org_compre4[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./pd96_org_slow_compression4_5step.png')

# pd_org_nmp1 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slowtrain_lessnmp/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp1_nhid96_nae2_nencdec1/pd.npy')
# pd_org_nmp2 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slowtrain_lessnmp/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp2_nhid96_nae2_nencdec1/pd.npy')
# pd_org_nmp3 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slowtrain_lessnmp/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd.npy')

# # plot_3_snapshots(gt[0,0,:,:,:,0],
# #                  gt[0,19,:,:,:,0],
# #                  gt[0,49,:,:,:,0],
# #                  gt[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./gt96new.png')

# plot_3_snapshots(pd_org_nmp1[0,0,:,:,:,0],
#                  pd_org_nmp1[0,19,:,:,:,0],
#                  pd_org_nmp1[0,49,:,:,:,0],
#                  pd_org_nmp1[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./pd96_org_slow_compression2_nmp1.png')


# plot_3_snapshots(pd_org_nmp2[0,0,:,:,:,0],
#                  pd_org_nmp2[0,19,:,:,:,0],
#                  pd_org_nmp2[0,49,:,:,:,0],
#                  pd_org_nmp2[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./pd96_org_slow_compression4_nmp2.png')


# plot_3_snapshots(pd_org_nmp3[0,0,:,:,:,0],
#                  pd_org_nmp3[0,19,:,:,:,0],
#                  pd_org_nmp3[0,49,:,:,:,0],
#                  pd_org_nmp3[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./pd96_org_slow_compression4_nmp3.png')



#"""correction of training and validation data, final pt4 is totally wrong"
# pt1 = np.load('/usr/WS2/tian9/KMC_3D_stats_long_slow/predict-3D-pt1.npy')
# pt2 = np.load('/usr/WS2/tian9/KMC_3D_stats_long_slow/predict-3D-pt2.npy')
# pt3 = np.load('/usr/WS2/tian9/KMC_3D_stats_long_slow/predict-3D-pt3.npy')
# pt4 = np.load('/usr/WS2/tian9/KMC_3D_stats_long_slow/predict-3D-pt4.npy')



# plot_3_snapshots(pt1[0,0,:,:,:,0],
#                  pt1[0,19,:,:,:,0],
#                  pt1[0,49,:,:,:,0],
#                  pt1[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./pt1.png')


# plot_3_snapshots(pt2[0,0,:,:,:,0],
#                  pt2[0,19,:,:,:,0],
#                  pt2[0,49,:,:,:,0],
#                  pt2[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./pt2.png')


# plot_3_snapshots(pt3[0,0,:,:,:,0],
#                  pt3[0,19,:,:,:,0],
#                  pt3[0,49,:,:,:,0],
#                  pt3[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./pt3.png')

# plot_3_snapshots(pt4[0,0,:,:,:,0],
#                  pt4[0,19,:,:,:,0],
#                  pt4[0,49,:,:,:,0],
#                  pt4[0,69,:,:,:,0],cmap = 'Greys_r',save_path ='./pt4.png')



gt = np.load('/usr/WS2/tian9/KMC_3D_slow_compress_corr/valid.npy')
# pt1 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# pt2 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd.npy')
# pt3 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# pt4 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd.npy')

pt5 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae3_nencdec1/pd.npy')
pt6 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae3_nencdec1/pd.npy')


# plot_3_snapshots(gt[0,0,:,:,:,0],
#                  gt[0,50,:,:,:,0],
#                  gt[0,100,:,:,:,0],
#                  gt[0,200,:,:,:,0],cmap = 'Greys_r',save_path ='./correct_96_visual/gt.png')

# plot_3_snapshots(pt1[0,0,:,:,:,0],
#                  pt1[0,50,:,:,:,0],
#                  pt1[0,100,:,:,:,0],
#                  pt1[0,200,:,:,:,0],cmap = 'Greys_r',save_path ='./correct_96_visual/nstep1ae1_long.png')


# plot_3_snapshots(pt2[0,0,:,:,:,0],
#                  pt2[0,50,:,:,:,0],
#                  pt2[0,100,:,:,:,0],
#                  pt2[0,200,:,:,:,0],cmap = 'Greys_r',save_path ='./correct_96_visual/nstep1ae2_long.png')


# plot_3_snapshots(pt3[0,0,:,:,:,0],
#                  pt3[0,50,:,:,:,0],
#                  pt3[0,100,:,:,:,0],
#                  pt3[0,200,:,:,:,0],cmap = 'Greys_r',save_path ='./correct_96_visual/nstep5ae1_long.png')

# plot_3_snapshots(pt4[0,0,:,:,:,0],
#                  pt4[0,50,:,:,:,0],
#                  pt4[0,100,:,:,:,0],
#                  pt4[0,200,:,:,:,0],cmap = 'Greys_r',save_path ='./correct_96_visual/nstep5ae2_long.png')

plot_3_snapshots(pt5[0,0,:,:,:,0],
                 pt5[0,10,:,:,:,0],
                 pt5[0,20,:,:,:,0],
                 pt5[0,30,:,:,:,0],cmap = 'Greys_r',save_path ='./correct_96_visual/nstep1ae3_long.png')

plot_3_snapshots(pt6[0,0,:,:,:,0],
                 pt6[0,10,:,:,:,0],
                 pt6[0,20,:,:,:,0],
                 pt6[0,30,:,:,:,0],cmap = 'Greys_r',save_path ='./correct_96_visual/nstep5ae3_long.png')