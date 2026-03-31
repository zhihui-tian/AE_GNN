source submit.sh

for delta in "" _nodelta; do
  if [ "$delta" == "_nodelta" ]; then DELTA=""; else DELTA=delta; fi
for patch in 1 4; do
for ker in 3 ; do
for nlayer in 1 4 ; do
for nhid in 64 128 256; do

hidden=${nhid}
for i in `seq 2 $nlayer`; do hidden=${hidden},${nhid}; done
#if (( $patch >= 4 )) || (( $ker >= 7 )) batch=
batch=16
echo "debug patch $patch ker $ker nlayer $nlayer nhid $nhid hidden $hidden batch=$batch"
tag=CHE-2d.i1_${nhid}x${nlayer}_k${ker}_p${patch}${delta}
sed "$(_opt sys=${tag} patch=${patch} model=predrnn hidden=${hidden} in=1 tot=17 step=2 pg=4m lr=1e-3 dim=2 shape=64 batch=${batch} periodic ker=${ker} $DELTA long)" job > tmp_scripts/${tag}.job

tag=CHE-2d.i1o1_${nhid}x${nlayer}_k${ker}_p${patch}${delta}
sed "$(_opt sys=${tag} patch=${patch} model=predrnn hidden=${hidden} in=1 tot=2 step=1 pg=4m lr=1e-3 dim=2 shape=64 batch=${batch} periodic ker=${ker} $DELTA long)" job > tmp_scripts/${tag}.job

tag=CHE-2d.convi1_${nhid}x${nlayer}_k${ker}_p${patch}${delta}
sed "$(_opt sys=${tag} patch=${patch} model=pure_conv hidden=${hidden} in=1 tot=17 step=2 pg=4m lr=1e-3 dim=2 shape=64 batch=${batch} periodic ker=${ker} $DELTA long)" job > tmp_scripts/${tag}.job

tag=CHE-2d.convi1o1_${nhid}x${nlayer}_k${ker}_p${patch}${delta}
sed "$(_opt sys=${tag} patch=${patch} model=pure_conv hidden=${hidden} in=1 tot=2 step=1 pg=4m lr=1e-3 dim=2 shape=64 batch=${batch} periodic ker=${ker} $DELTA long)" job > tmp_scripts/${tag}.job

tag=CHE-2d.convi3_${nhid}x${nlayer}_k${ker}_p${patch}${delta}
sed "$(_opt sys=${tag} patch=${patch} model=pure_conv hidden=${hidden} in=3 tot=17 step=3 pg=4m lr=1e-3 dim=2 shape=64 batch=${batch} periodic ker=${ker} $DELTA long)" job > tmp_scripts/${tag}.job

done
done
done
done
done






for delta in "" _nodelta; do
  if [ "$delta" == "_nodelta" ]; then DELTA=""; else DELTA=delta; fi
for patch in 1 2 4; do
for ker in 3 ; do
for nlayer in 1 2 4 ; do
for nhid in 64 128; do

hidden=${nhid}
for i in `seq 2 $nlayer`; do hidden=${hidden},${nhid}; done
#if (( $patch >= 4 )) || (( $ker >= 7 )) batch=
batch=16
echo "debug patch $patch ker $ker nlayer $nlayer nhid $nhid hidden $hidden batch=$batch"
tag=CHE-2d.i1o16conv_${nhid}x${nlayer}_k${ker}_p${patch}${delta}

sed "$(_opt sys=${tag} patch=${patch} model=pure_conv hidden=${hidden} in=1 tot=17 step=2 pg=4m lr=1e-3 dim=2 shape=64 batch=${batch} periodic ker=${ker} $DELTA long)" job > tmp_scripts/${tag}.job
# |diff - job

done
done
done
done
done






for delta in "" _nodelta; do
  if [ "$delta" == "_nodelta" ]; then DELTA=""; else DELTA=delta; fi
for patch in 1 2 4; do
for ker in 3 ; do
for nlayer in 1 2 4 ; do
for nhid in 64 128; do

hidden=${nhid}
for i in `seq 2 $nlayer`; do hidden=${hidden},${nhid}; done
#if (( $patch >= 4 )) || (( $ker >= 7 )) batch=
batch=16
echo "debug patch $patch ker $ker nlayer $nlayer nhid $nhid hidden $hidden batch=$batch"
tag=CHE-2d.i1conv_${nhid}x${nlayer}_k${ker}_p${patch}${delta}

sed "$(_opt sys=${tag} patch=${patch} model=pure_conv hidden=${hidden} in=1 tot=2 step=1 pg=4m lr=1e-3 dim=2 shape=64 batch=${batch} periodic ker=${ker} $DELTA long)" job > tmp_scripts/${tag}.job
# |diff - job

done
done
done
done
done

#for j in tmp_scripts/*k[2-9]*.job; do
#bsub $j; 
#sleep 7300
#done

#exit
for delta in "" _nodelta; do
  if [ "$delta" == "_nodelta" ]; then DELTA=False; else DELTA=True; fi
for patch in 1 2 4; do
for ker in 1 3 5 7; do
for nlayer in 1 2 3 4 5; do
for nhid in 32 64 128 256; do

hidden=${nhid}
for i in `seq 2 $nlayer`; do hidden=${hidden},${nhid}; done
#if (( $patch >= 4 )) || (( $ker >= 7 )) batch=
batch=8
echo "debug patch $patch ker $ker nlayer $nlayer nhid $nhid hidden $hidden batch=$batch"
tag=CHE-2d.i3_${nhid}x${nlayer}_k${ker}_p${patch}${delta}

sed "$(_opt sys=${tag} patch=${patch} model=predrnn hidden=${hidden} in=3 tot=17 step=3 pg=4m lr=1e-3 dim=2 shape=64 batch=${batch} periodic ker=${ker} )" job |sed "s/fit_delta [A-z]*/fit_delta $DELTA/" > tmp_scripts/${tag}.job
# |diff - job

done
done
done
done
done

/bin/ls tmp_scripts/CHE-2d.i3_*x[1-4]*k[1-5]_*a.job  | parallel -j1  bsub -G ustruct -W 12:00 -q pbatch

/bin/ls tmp_scripts/CHE-2d.i3_*x[1-4]*k[1-5]_*[0-9].job  | parallel -j1 bsub -G spnflcgc -W 12:00 -q pbatch 
