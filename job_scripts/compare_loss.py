# import torch
# import matplotlib.pyplot as plt
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'DejaVu Serif'],
#     'mathtext.fontset': 'dejavuserif',
# })


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

# # x = torch.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/silu_l1/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL1_nmp3_nhid96_nae1_nencdec1/loss.pt', map_location='cpu')
# 
# print(type(x))
# print(x.keys() if isinstance(x, dict) else x)



# x = torch.load('/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/silu_l1/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL1_nmp3_nhid96_nae1_nencdec1/loss_log.pt', map_location='cpu')
# print(type(x), len(x))
# for i, item in enumerate(x):
#     print(f"Item {i}: type={type(item)}")
#     if hasattr(item, 'shape'):
#         print(" shape:", item.shape)
#     elif isinstance(item, (list, tuple)):
#         print(" length:", len(item))


# for nmp in [3,5,12,16]:
#     train_loss, val_loss = torch.load(f'/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/silu_l1/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL1_nmp{nmp}_nhid96_nae1_nencdec1/loss_log.pt', map_location='cpu')
#     plt.plot(train_loss, label=f'train_nmp{nmp}')
#     plt.plot(val_loss, label=f'val_nmp{nmp}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
# plt.show()
# plt.savefig('silu_l1.png')
# plt.close()



import torch
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4), dpi=200)

# 颜色：同 nmp 同色
colors = {
    3: "#1f77b4",   # 蓝
    5: "#ff7f0e",   # 橙
    12: "#2ca02c",  # 绿
    16: "#d62728",  # 红
}

for nmp in [3, 5, 12, 16]:
# for nmp in [3, 5]:
    path = f"/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/60epoch/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp{nmp}_nhid96_nae1_nencdec1/loss_log.pt"
    train_loss, val_loss = torch.load(path, map_location="cpu")
    c = colors[nmp]

    # Train: 虚线+圆点
    plt.plot(train_loss[:50], linestyle="--", marker="o", color=c,
             label=f"Train nmp={nmp}", markersize=1, linewidth=0.8)
    # Val: 实线+方块
    plt.plot(val_loss[:50], linestyle="-", marker="s", color=c,
             label=f"Val nmp={nmp}", markersize=1, linewidth=1)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title('relu')
# plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
plt.legend(fontsize=12, ncol=2, frameon=False)
plt.tight_layout()

plt.savefig("relu_l2.png", dpi=300)
# plt.show()



import torch
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4), dpi=200)

# 颜色：同 nmp 同色
colors = {
    3: "#1f77b4",   # 蓝
    5: "#ff7f0e",   # 橙
    12: "#2ca02c",  # 绿
    16: "#d62728",  # 红
}

for nmp in [3, 5, 12, 16]:
# for nmp in [3, 5]:
    path = f'/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/silu_l1/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL1_nmp{nmp}_nhid96_nae1_nencdec1/loss_log.pt'
    train_loss, val_loss = torch.load(path, map_location="cpu")
    c = colors[nmp]

    # Train: 虚线+圆点
    plt.plot(train_loss[:50], linestyle="--", marker="o", color=c,
             label=f"Train nmp={nmp}", markersize=1, linewidth=0.8)
    # Val: 实线+方块
    plt.plot(val_loss[:50], linestyle="-", marker="s", color=c,
             label=f"Val nmp={nmp}", markersize=1, linewidth=1)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title('silu')
# plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
plt.legend(fontsize=12, ncol=2, frameon=False)
plt.tight_layout()

plt.savefig("silu_l1.png", dpi=300)
# plt.show()



import torch
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4), dpi=200)

# 颜色：同 nmp 同色
colors = {
    3: "#1f77b4",   # 蓝
    5: "#ff7f0e",   # 橙
    12: "#2ca02c",  # 绿
    16: "#d62728",  # 红
}

for nmp in [3, 5, 12, 16]:

# for nmp in [3, 5]:
    path = f'/usr/WS2/tian9/tian9/kmc_gnn/micro/NPS-versions/NPS_tuolu3d/experiment/silu_l2/grain_NPS_autoencoder-batch4_lr1e-4_nin1_nout5_noise1e-3_lossL2_nmp{nmp}_nhid96_nae1_nencdec1/loss_log.pt'
    train_loss, val_loss = torch.load(path, map_location="cpu")
    c = colors[nmp]

    # Train: 虚线+圆点
    plt.plot(train_loss[:50], linestyle="--", marker="o", color=c,
             label=f"Train nmp={nmp}", markersize=1, linewidth=0.8)
    # Val: 实线+方块
    plt.plot(val_loss[:50], linestyle="-", marker="s", color=c,
             label=f"Val nmp={nmp}", markersize=1, linewidth=1)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title('silu')
# plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
plt.legend(fontsize=12, ncol=2, frameon=False)
plt.tight_layout()

plt.savefig("silu_l2.png", dpi=300)
# plt.show()


