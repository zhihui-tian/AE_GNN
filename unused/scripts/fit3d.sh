#sed 's/_TEST/0/;s/_SYS/waves-3d.i7predrnn/;s/e3d_lstm/predrnn/;s/_IN_LEN/5/;s/_TOT_LEN/17/;s/_CLIP_STEP/5/;s/_CLIP_LEN/-1/;s/_LR/1e-3 --dim 3 --img_shape 32,32,32/;s/--num_hidden[ ,0-9]*/--num_hidden=128,128,128,128/;s/BSUB -q.*/BSUB -q pbatch/;s/BSUB -W.*/BSUB -W 12:00/;' job 

sed "$(_opt sys=waves-3d.i7predrnn_pp model=predrnn_pp hidden=128,64,64,64 in=7 tot=17 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 long)" job|bsub
sed "$(_opt sys=waves-3d.i7predrnn model=predrnn hidden=128,128,128,128 in=7 tot=17 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 long)" job|bsub
sed "$(_opt sys=waves-3d.i7predrnn-p2 model=predrnn hidden=128,128,128,128 patch=2 in=7 tot=17 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 patch=2 long)" job|bsub

sed "$(_opt sys=ball-3d.i7predrnn_pp model=predrnn_pp hidden=128,64,64,64 in=7 tot=17 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 long)" job|bsub
sed "$(_opt sys=ball-3d.i7predrnn model=predrnn hidden=128,128,128,128 in=7 tot=17 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 long)" job|bsub
sed "$(_opt sys=ball-3d.i7predrnn-p2 model=predrnn hidden=128,128,128,128 patch=2 in=7 tot=17 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 long)" job|bsub

sed "$(_opt sys=CHE-3d.i7predrnn_pp model=predrnn_pp hidden=128,64,64,64 in=7 tot=17 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 long)" job|bsub
sed "$(_opt sys=CHE-3d.i7predrnn-p2 model=predrnn hidden=128,128,128,128 patch=2 in=7 tot=17 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 long)" job|bsub
sed "$(_opt sys=CHE-3d.i7predrnn model=predrnn hidden=128,128,128,128 in=7 tot=17 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 long)" job|bsub

sed "$(_opt sys=CHE-3d.peri7predrnn model=predrnn hidden=128,128,128,128 in=7 tot=17 step=6 pg=Oh lr=1e-3 dim=3 shape=32,32,32 batch=2 periodic long)" job|bsub

sed "$(_opt sys=waves-3dperiodic.i7predrnn model=predrnn hidden=128,128,128,128 in=7 tot=17 step=6 pg=Oh lr=1e-3 dim=3 shape=32,32,32 batch=2 periodic long)" job|bsub

sed "$(_opt sys=abv128.i5predrnn model=predrnn hidden=128,128,128,128 in=5 tot=15 step=6 pg=Oh lr=1e-3 dim=3 shape=64,64,64 batch=1 periodic long)" job|bsub

sed "$(_opt sys=LJ32.i5predrnn model=predrnn hidden=128,128,128,128 in=5 tot=15 step=6 pg=Oh lr=1e-3 dim=3 shape=32,32,32 batch=2 periodic long)" job|bsub


sed "$(_opt sys=grain-Al.i5k3predrnn_nodelta model=predrnn hidden=64,64,64,64 in=5 tot=15 step=6 pg=Oh lr=1e-3 dim=3 shape=32,32,32 ker=3 patch=2 batch=2 periodic long)" job |bsub

sed "$(_opt sys=ball-3d.i5predrnn_nodelta  model=predrnn hidden=64,64,64,64 in=5 tot=15 clip=-1 step=6 pg=1 lr=1e-3 dim=3 shape=32,32,32 patch=2 ker=3 long)" job|bsub

sed "$(_opt sys=LJ32.i5predrnn-k3p2h64 model=predrnn hidden=64,64,64,64 in=5 tot=15 step=6 pg=Oh lr=1e-3 dim=3 shape=32,32,32 batch=4 ker=3 patch=2 periodic long)" job|bsub

sed "$(_opt sys=heat.heat model=feedforward net=heat_eq hidden=64 in=1 tot=2 step=1 pg=1 lr=1e-3 patch=1 batch=1 ker=3 dim=3 shape=20 periodic delta noputback)" job

sed "$(_opt sys=LJ25.cheL1 loss='1*L1' model=feedforward net=cahn_hilliard hidden=128 in=1 tot=2 step=2 pg=Oh lr=1e-3 patch=1 batch=16 ker=3 dim=3 shape=25 periodic delta noputback)" job |bsub

