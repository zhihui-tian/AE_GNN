
import numpy as np
import matplotlib.pyplot as plt


gt = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')
gnn_only = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS-main/experiment/grain_NPS-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')

ae1 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae1_nencdec1/pd.npy')
ae2 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae2_nencdec1/pd.npy')
ae3 = np.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout1_noise1e-3_lossL2_nmp3_nhid96_nae3_nencdec1/pd.npy')

et = np.load('/usr/WS2/tian9/ethanresult/NPS-runs/KMC_3D_2_4_bh_slow_grain2d_NPS_autoencoder/batch8_lr1e-3_nin1_nout5-noiseadd_normal5e-1_lossL1_nmp16_nae2_nencdec2_dropout0/pd.npy')

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
rmse_all_gnn = []
rmse_all_ae1 = []
rmse_all_ae2 = []
rmse_all_ae3 = []
rmse_all_et = []
for i in range(20):
    rmse_all_gnn.append(rmse(gt[0,i],gnn_only[0,i]))
    rmse_all_ae1.append(rmse(gt[0,i],ae1[0,i]))
    rmse_all_ae2.append(rmse(gt[0,i],ae2[0,i]))
    rmse_all_ae3.append(rmse(gt[0,i],ae3[0,i]))
    rmse_all_et.append(rmse(gt[0,i],et[0,i]))


fig, ax = plt.subplots(figsize=(10, 8.5))
ax.plot(np.arange(20),rmse_all_gnn,label = 'GNN Baseline')
ax.plot(np.arange(20),rmse_all_ae1,label = 'Linear Compression Ratio 2')
ax.plot(np.arange(20),rmse_all_ae2,label = 'Linear Compression Ratio 4')
ax.plot(np.arange(20),rmse_all_ae3,label = 'Linear Compression Ratio 8')
ax.plot(np.arange(20),rmse_all_et,label = 'ethan result')
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
fig.savefig("rmse_ae3d.svg", format='svg',bbox_inches='tight',pad_inches=0.1)