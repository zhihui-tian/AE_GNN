import numpy as np
import matplotlib.pyplot as plt


from matplotlib import font_manager
import matplotlib.pyplot as plt
font_manager.fontManager.addfont("/g/g92/tian9/.local/share/fonts/times.ttf")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
})

plt.rcParams.update({
    'font.size': 12,              # 全局字体大小
    'axes.titlesize': 12,         # 坐标轴标题字体
    'axes.labelsize': 12,         # 坐标轴标签字体
    'xtick.labelsize': 12,        # x 轴刻度字体
    'ytick.labelsize': 12,        # y 轴刻度字体
    'legend.fontsize': 12,        # 图例字体
})



"""figure for statistical prediction validation in 3D, figure2f"""
####################################################################################### 32^3 #######################################################################################
gt = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')
# gnn_only = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS-main/experiment/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
gnn_only = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/spatial_32_out1/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')

ae1 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/spatial_32_out1/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd_32.npy')
# ae2 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/spatial_32_out1/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd_32.npy')
# ae3 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/spatial_32_out1/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae3_nencdec1/pd_32.npy')

ae4 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/steps_tune/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout3_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
ae5 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/steps_tune/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
ae7 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/steps_tune/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout7_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
rmse_all_gnn = []
rmse_all_ae1 = []
# rmse_all_ae2 = []
# rmse_all_ae3 = []
rmse_all_ae4 = []
rmse_all_ae5 = []
rmse_all_ae7 = []
for i in range(20):
    rmse_all_gnn.append(rmse(gt[0,i],gnn_only[0,i]))
    rmse_all_ae1.append(rmse(gt[0,i],ae1[0,i]))
    # rmse_all_ae2.append(rmse(gt[0,i],ae2[0,i]))
    # rmse_all_ae3.append(rmse(gt[0,i],ae3[0,i]))
    rmse_all_ae4.append(rmse(gt[0,i],ae4[0,i]))
    rmse_all_ae5.append(rmse(gt[0,i],ae5[0,i]))
    rmse_all_ae7.append(rmse(gt[0,i],ae7[0,i]))



fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(np.arange(20),rmse_all_gnn,label = 'GNN Baseline')

ax.plot(np.arange(20),rmse_all_ae1,label = 'AE+GNN 1-step')
# ax.plot(np.arange(20),rmse_all_ae2,label = 'Linear Compression Ratio 4')
# ax.plot(np.arange(20),rmse_all_ae3,label = 'Linear Compression Ratio 8')
ax.plot(np.arange(20),rmse_all_ae4,label = 'AE+GNN 3-step')
ax.plot(np.arange(20),rmse_all_ae5,label = 'AE+GNN 5-step')
ax.plot(np.arange(20),rmse_all_ae7,label = 'AE+GNN 7-step')
# Formatting
ax.set_ylabel('RMSE')
ax.set_xlabel('Simulation Step', fontsize=13)
ax.set_xticks(np.arange(0, 20, 5))
ax.grid(False)

# Layout and title
fig.subplots_adjust(bottom=0.26)
ax.legend(loc='best', fontsize=12, frameon=False)
fig.text(0.5, 0.11, 'Figure 2(f): 3D RMSE Comparisonn', ha='center', fontsize=13)
# Save
fig.savefig("rmse_ae3d_tnr.svg", format='svg',bbox_inches='tight',pad_inches=0.1)






####################################################################################### 96^3 #######################################################################################
# gt = np.load('/usr/WS2/tian9/KMC_3D_stats_long_slow/valid.npy')
# nmp1 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slowtrain_lessnmp/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp1_nhid96_nae2_nencdec1/pd.npy')
# nmp2 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slowtrain_lessnmp/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp2_nhid96_nae2_nencdec1/pd.npy')
# nmp3 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slowtrain_lessnmp/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd.npy')

# nmp3_1step = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slowtrain_moreepoch/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd.npy')
# def rmse(y_true, y_pred):
#     return np.sqrt(np.mean((y_true - y_pred) ** 2))
# rmse_all_nmp1 = []
# rmse_all_nmp2 = []
# rmse_all_nmp3 = []
# rmse_all_nmp3_1step=[] 
# for i in range(20):
#     rmse_all_nmp1.append(rmse(gt[0,i],nmp1[0,i]))
#     rmse_all_nmp2.append(rmse(gt[0,i],nmp2[0,i]))
#     rmse_all_nmp3.append(rmse(gt[0,i],nmp3[0,i]))
#     rmse_all_nmp3_1step.append(rmse(gt[0,i],nmp3_1step[0,i]))




# fig, ax = plt.subplots(figsize=(8, 5.5))
# ax.plot(np.arange(20),rmse_all_nmp1,label = '1 MP 5 step')
# ax.plot(np.arange(20),rmse_all_nmp2,label = '2 MP 5 step')
# ax.plot(np.arange(20),rmse_all_nmp3,label = '3 MP 5 step')
# ax.plot(np.arange(20),rmse_all_nmp3_1step,label = '3 MP 1 step')

# # Formatting
# ax.set_ylabel('RMSE')
# ax.set_xlabel('Simulation Step', fontsize=13)
# ax.set_xticks(np.arange(0, 20, 5))
# ax.grid(False)

# # Layout and title
# fig.subplots_adjust(bottom=0.26)
# ax.legend(loc='best', fontsize=12, frameon=False)
# fig.text(0.5, 0.11, 'figure less mp for high compression', ha='center', fontsize=13)
# # Save
# fig.savefig("less_mp_high_compression.svg", format='svg',bbox_inches='tight',pad_inches=0.1)



### fig 8
# gt = np.load('/usr/WS2/tian9/KMC_3D_slow_compress_corr/valid.npy')
# step1_ae1 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/archived/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# step5_ae1 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/archived/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
# step1_ae2 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/archived/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd.npy')
# step5_ae2 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/high_compression_slow_splitdata/archived/slow5_step/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd.npy')

# def rmse(y_true, y_pred):
#     return np.sqrt(np.mean((y_true - y_pred) ** 2))
# rmse_all_step1_ae1 = []
# rmse_all_step5_ae1 = []
# rmse_all_step1_ae2 = []
# rmse_all_step5_ae2 = []

# for i in range(75):
#     rmse_all_step1_ae1.append(rmse(gt[0,i],step1_ae1[0,i]))
#     rmse_all_step5_ae1.append(rmse(gt[0,i],step5_ae1[0,i]))
#     rmse_all_step1_ae2.append(rmse(gt[0,i],step1_ae2[0,i]))
#     rmse_all_step5_ae2.append(rmse(gt[0,i],step5_ae2[0,i]))





# fig, ax = plt.subplots(figsize=(8, 5.5))
# ax.plot(np.arange(75),rmse_all_step1_ae1,label = 'AE+GNN(n=2, 1-step)')
# ax.plot(np.arange(75),rmse_all_step5_ae1,label = 'AE+GNN(n=2, 5-step)')
# ax.plot(np.arange(75),rmse_all_step1_ae2,label = 'AE+GNN(n=4, 1-step)')
# ax.plot(np.arange(75),rmse_all_step5_ae2,label = 'AE+GNN(n=4, 5-step)')


# # Formatting
# ax.set_ylabel('RMSE')
# ax.set_xlabel('Simulation Step', fontsize=13)
# ax.set_xticks(np.arange(0, 75, 25))
# ax.grid(False)

# # Layout and title
# fig.subplots_adjust(bottom=0.26)
# ax.legend(loc='best', fontsize=12, frameon=False)
# fig.text(0.5, 0.11, 'figure less mp for high compression', ha='center', fontsize=13)
# # Save
# fig.savefig("steps_high_compression_tnr.svg", format='svg',bbox_inches='tight',pad_inches=0.1)





































# gt = np.load('/usr/WS2/tian9/KMC_3D_stats_long_slow/valid.npy')
# gnn_only=np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/GNN_only_relu/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd96.npy')
# pd_org = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/60epoch/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd96_org_slow.npy')
# def rmse(y_true, y_pred):
#     return np.sqrt(np.mean((y_true - y_pred) ** 2))

# gt_average=[]
# gnn_var=[]
# rmse_all_gnn_only = []
# rmse_all_aegnn = []

# for i in range(100):
#     # gt_average.append(np.std(gt[0,i]))
#     gnn_var.append(np.std(gnn_only[0,i]))

#     rmse_all_gnn_only.append(rmse(gt[0,i],gnn_only[0,i]))
#     rmse_all_aegnn.append(rmse(gt[0,i],pd_org[0,i]))




# fig, ax = plt.subplots(figsize=(8, 5.5))


# # ax.plot(np.arange(100),gt_average,label = 'gt_average')
# ax.plot(np.arange(100),gnn_average,label = 'GNN variance')
# ax.plot(np.arange(100),rmse_all_gnn_only,label = 'GNN Baseline')
# ax.plot(np.arange(100),rmse_all_aegnn,label = 'AE+GNN')

# # Formatting
# ax.set_ylabel('RMSE')
# ax.set_xlabel('Simulation Step', fontsize=13)
# ax.set_xticks(np.arange(0, 100, 5))
# ax.grid(False)

# # Layout and title
# fig.subplots_adjust(bottom=0.26)
# ax.legend(loc='best', fontsize=12, frameon=False)
# fig.text(0.5, 0.11, 'Figure 2(f): 3D RMSE Comparisonn', ha='center', fontsize=13)
# # Save
# fig.savefig("rmse_96_3.svg", format='svg',bbox_inches='tight',pad_inches=0.1)

# steps = np.arange(100)
# ax.plot(steps, rmse_all_gnn_only, color='orange', label='GNN Baseline')
# ax.plot(steps, rmse_all_aegnn, color='green', label='AE+GNN')
# ax.set_xlabel('Simulation Step')
# ax.set_ylabel('RMSE', color='black')
# ax.tick_params(axis='y', labelcolor='black')


# # ---- 右轴：Variance ----
# ax2 = ax.twinx()
# ax2.plot(steps, gnn_var, color='blue', label='GNN variance')
# ax2.set_ylabel('Variance', color='black')
# ax2.tick_params(axis='y', labelcolor='black')

# ax.legend(loc='upper left', title='RMSE')
# ax2.legend(loc='upper right', title='Variance')
# # ---- 合并图例 ----
# lines, labels = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax.legend(lines + lines2, labels + labels2, loc='best')

# # ---- Caption 模拟 ----
# plt.figtext(0.5, -0.05, 'Figure 2(f): 3D RMSE and Variance Comparison', 
#             wrap=True, ha='center', fontsize=10)
# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(7.2, 5.0))

# steps = np.arange(100)
# ax.plot(steps, rmse_all_gnn_only, color='orange', label='GNN Baseline')
# ax.plot(steps, rmse_all_aegnn,  color='green',  label='AE+GNN')
# ax.set_xlabel('Simulation Step')
# ax.set_ylabel('RMSE', color='black')
# ax.tick_params(axis='y', labelcolor='black')

# # 右轴
# ax2 = ax.twinx()
# ax2.plot(steps, gnn_var, color='blue', label='GNN variance')
# ax2.set_ylabel('Variance', color='blue', fontsize=12)
# ax2.tick_params(axis='y', colors='blue')
# ax2.spines['right'].set_color('blue')

# leg1 = ax.legend(loc='upper left', title='RMSE')
# plt.setp(leg1.get_title(), color='black')  # Legend title 颜色与左轴一致

# # 右边 legend（Variance 组）
# leg2 = ax2.legend(loc='upper right', title='Variance')
# plt.setp(leg2.get_title(), color='blue')  # Legend title 颜色与右轴一致

# # ---- Caption ----
# plt.figtext(0.5, -0.05, 'Figure 2(f): 3D RMSE and Variance Comparison',
#             ha='center', fontsize=11)

# fig.savefig("rmse_96_31.svg", format='svg',bbox_inches='tight',pad_inches=0.1)