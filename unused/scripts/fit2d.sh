
sed "$(_opt sys=dendrite_j6all.i12 in=12 tot=22 clip=100 step=10 pg=4m lr=1e-3 long)" job|bsub
sed "$(_opt sys=dendrite_j6all.i12predrnn model=predrnn hidden=128,128,128,128 in=12 tot=22 clip=100 step=10 pg=4m lr=1e-3 long)" job|bsub
sed "$(_opt sys=dendrite_j6all.i12predrnn_pp model=predrnn_pp hidden=128,64,64,64 in=12 tot=22 clip=100 step=10 pg=4m lr=1e-3 long)" job|bsub

sed "$(_opt sys=waves-1mode.i8e3d model=e3d_lstm in=8 tot=18 clip=200 step=10 pg=4m lr=1e-3 long)" job|bsub
sed "$(_opt sys=waves-1mode.i8predrnn model=predrnn hidden=128,128,128,128 in=8 tot=18 clip=200 step=10 pg=4m lr=1e-3 long)" job|bsub
sed "$(_opt sys=waves-1mode.i8predrnn_pp model=predrnn_pp hidden=128,64,64,64 in=8 tot=18 clip=200 step=10 pg=4m lr=1e-3 long)" job|bsub

sed "$(_opt sys=cahn-hilliard-SOC-third.i10pg model=e3d_lstm hidden=64,64,64,64 in=10 tot=20 clip=100 step=10 pg=4m lr=1e-3 )" job|bsub
sed "$(_opt sys=cahn-hilliard-SOC-third.i10predrnn model=predrnn hidden=128,128,128,128 in=10 tot=20 clip=100 step=10 pg=4m lr=1e-3 )" job|bsub
sed "$(_opt sys=cahn-hilliard-SOC-third.i10predrnn_pp model=predrnn_pp hidden=128,64,64,64 in=10 tot=20 clip=100 step=10 pg=4m lr=1e-3 )" job|bsub

sed "$(_opt sys=crack2.i10e3d model=e3d_lstm hidden=64,64,64,64 in=10 tot=20 clip=150 step=5 pg=mm lr=1e-3 long)" job|bsub
sed "$(_opt sys=crack2.i10predrnn model=predrnn hidden=128,128,128,128 batch=3 in=10 tot=20 clip=150 step=5 pg=mm lr=1e-3 long)" job|bsub
sed "$(_opt sys=crack2.i10predrnn_pp model=predrnn_pp hidden=128,64,64,64 in=10 tot=20 clip=150 step=5 pg=mm lr=1e-3 long)" job|bsub

sed "$(_opt sys=moving-mnist-example.i10e3d model=e3d_lstm hidden=64,64,64,64 in=10 tot=20 clip=20 step=5 pg=1 lr=1e-3 long)" job|bsub
sed "$(_opt sys=moving-mnist-example.i10predrnn model=predrnn hidden=128,128,128,128 batch=3 in=10 tot=20 clip=20 step=5 pg=1 lr=1e-3 long)" job|bsub
sed "$(_opt sys=moving-mnist-example.i10predrnn_pp model=predrnn_pp hidden=128,64,64,64 in=10 tot=20 clip=20 step=5 pg=1 lr=1e-3 long)" job|bsub

sed "$(_opt sys=waves-2mode.i10e3d model=e3d_lstm hidden=64,64,64,64 in=10 tot=20 clip=200 step=5 pg=4m lr=1e-3 long)" job|bsub
sed "$(_opt sys=waves-2mode.i10predrnn model=predrnn hidden=128,128,128,128 batch=3 in=10 tot=20 clip=200 step=5 pg=4m lr=1e-3 long)" job|bsub
sed "$(_opt sys=waves-2mode.i10predrnn_pp model=predrnn_pp hidden=128,64,64,64 in=10 tot=20 clip=200 step=5 pg=4m lr=1e-3 long)" job|bsub

sed "$(_opt sys=dendrite_j6all.i12-n50  in=12 tot=22 step=10 pg=4m lr=1e-3 saveall nclip=50 long)" job|bsub
sed "$(_opt sys=dendrite_j6all.i12-n100  in=12 tot=22 step=10 pg=4m lr=1e-3 saveall nclip=100 long)" job|bsub
sed "$(_opt sys=dendrite_j6all.i12-n200  in=12 tot=22 step=10 pg=4m lr=1e-3 saveall nclip=200 long)" job|bsub
sed "$(_opt sys=dendrite_j6all.i12-n400  in=12 tot=22 step=10 pg=4m lr=1e-3 saveall nclip=400 long)" job|bsub

sed "$(_opt sys=cahn-hilliard-SOC-third.i10-n50 in=10 tot=20 step=10 pg=4m lr=1e-3 saveall nclip=50 long)" job|bsub
sed "$(_opt sys=cahn-hilliard-SOC-third.i10-n100 in=10 tot=20 step=10 pg=4m lr=1e-3 saveall nclip=100 long)" job|bsub
sed "$(_opt sys=cahn-hilliard-SOC-third.i10-n200 in=10 tot=20 step=10 pg=4m lr=1e-3 saveall nclip=200 long)" job|bsub
sed "$(_opt sys=cahn-hilliard-SOC-third.i10-n400 in=10 tot=20 step=10 pg=4m lr=1e-3 saveall nclip=400 long)" job|bsub
sed "$(_opt sys=cahn-hilliard-SOC-third.i10-n552 in=10 tot=20 step=10 pg=4m lr=1e-3 saveall nclip=552 long)" job|bsub

sed "$(_opt sys=waves-1mode.i8-n50 in=8 tot=18 step=10 pg=4m lr=1e-3 saveall nclip=50 long)" job|bsub
sed "$(_opt sys=waves-1mode.i8-n100 in=8 tot=18 step=10 pg=4m lr=1e-3 saveall nclip=100 long)" job|bsub
sed "$(_opt sys=waves-1mode.i8-n200 in=8 tot=18 step=10 pg=4m lr=1e-3 saveall nclip=200 long)" job|bsub
sed "$(_opt sys=waves-1mode.i8-n400 in=8 tot=18 step=10 pg=4m lr=1e-3 saveall nclip=400 long)" job|bsub

sed "$(_opt sys=CHE-2dfine.che model=feedforward hidden=3 in=1 tot=50 step=10 pg=4m lr=1e-3 patch=1 batch=8 ker=5 dim=2 shape=1024,1024 periodic)" job |bsub
sed "$(_opt sys=laplacian_sq.conv_ker5_hid1 model=pure_conv hidden=1 in=1 tot=10 step=5 pg=4m lr=1e-3 patch=1 batch=16 ker=5 dim=2 shape=64,64 periodic)" job |bsub

sed "$(_opt sys=CHE-2dfine.che model=feedforward net=cahn_hilliard hidden=64 in=1 tot=2 step=2 pg=4m lr=1e-3 patch=1 batch=16 ker=3 dim=2 shape=64 periodic delta)"  job  |bsub

sed "$(_opt sys=che2d-perFeb2020.i2k3predrnn model=predrnn hidden=64,64,64,64 in=5 tot=15 step=4 pg=4m lr=1e-3 dim=2 shape=64 batch=4 ker=3 patch=1 periodic long)" job|bsub

sed "$(_opt sys=X-ray-defects.resnet1 model=feedforward net=resnet hidden=64 in=1 tot=2 step=4 pg=1 lr=1e-3 dim=2 shape=1024 batch=1 ker=3 patch=1 loss=1*L1)" job |bsub

sed "$(_opt sys=che2d-perFeb2020.i2k3e3d model=e3d_lstm hidden=64,64,64,64 in=5 tot=15 step=3 pg=4m lr=1e-3 dim=2 shape=64 batch=2 ker=3 patch=1 periodic long)" job|bsub

