#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 train|test TSKIP NOISE"
    exit
fi
job=$1
TSKIP=$2
NOISE=$3

if [[ $job == 'train' ]]; then
CLIP_LEN=20
SIMLEN=$((TSKIP*CLIP_LEN))
rm -f c*.npy
for begin in 400 800 1600 2400 3200 4000 4800 5600 7200 8800; do
    end=$((begin+SIMLEN))
    for crange in "-0.6 -0.5" "-0.5 0" "0 0.5" "0.5 0.6"; do
        IFS=' ' read -r -a array <<< "$crange"
        cmin="${array[0]}"
        cmax="${array[1]}"
        tag=c_${cmin}_${cmax}_begin${begin}
        tags=`for i in {00..73}; do echo -n ${tag}_${i} " "; done`
        parallel python generate_SPDE.py --data_size "'1 $end 64 64'" --method che --D 0.01 --tskip $TSKIP --tskip_noise $TSKIP --tbegin $begin --noise $NOISE --cmin $cmin --cmax $cmax  -o ::: $tags
    done
done
gather-dat.py c*[0-6]?.npy --cat -o train.npy
gather-dat.py c*_[7]?.npy --cat -o valid.npy
rm -f c*.npy
# for noiseless dataset, add initial configs from noisy runs
#python calculator_array.py ' /g/g90/zhou6/lassen-space/data/che2dp_noise/step32/train.npy [:,0,...,0]'  -o c0.npy
#TSKIP=32
#python generate_SPDE.py --data_size "2800 640 64 64" --method che --D 0.01 --tskip $TSKIP --tskip_noise $TSKIP --tbegin 0 --noise 0 --c0 c0.npy -o train.npy
# alternatively
#calculator_array.py ' [np.save("c0_%d"%i, a[None,:]) for i,a in enumerate( c0.npy )]'
#seq 0 23 | parallel -I% --max-args 1 python generate_SPDE.py --data_size '"1 12800 64 64"' --method che --D 0.01 --tskip 32 --noise 0.01 --c0 c0_%.npy  -o out_%

# ./SPDE-train_test_dataset.sh train 1 0.01
#tags=`echo {00..30}`
#parallel python generate_SPDE.py --data_size "'100 20 64 64'" --method che --D 0.01 --tskip 1 --tskip_noise 1 --tbegin 0 --noise 0.01 --cmin -0.7 --cmax 0.7 --dc0 0.7  -o ::: $tags
#gather-dat.py ??.npy --cat

# ./SPDE-train_test_dataset.sh test 32 0.01
elif [[ $job == 'test' ]]; then 
CLIP_LEN=600
SIMLEN=$((TSKIP*CLIP_LEN))
rm -f c*.npy
for begin in 1000 6000; do
    end=$((begin+SIMLEN))
    for crange in "-0.6 -0.577" "-0.55 0.55" "0.577 0.6"; do
        IFS=' ' read -r -a array <<< "$crange"
        cmin="${array[0]}"
        cmax="${array[1]}"
        tag=c_${cmin}_${cmax}_begin${begin}
        tags=`for i in {00..02}; do echo -n ${tag}_${i} " "; done`
        parallel python generate_SPDE.py --data_size "'1 $end 64 64'" --method che --D 0.01 --tskip $TSKIP --tskip_noise 1 --tbegin $begin --noise $NOISE --cmin $cmin --cmax $cmax  -o ::: $tags
    done
done
gather-dat.py c*.npy --cat -o test.npy
rm -f c*.npy
fi

# if false; then
# # with noise training set
# for begin in 400 800 1600 2400 3200 4000 4800 5600 7200 8800; do
#     end=$((begin+CLIPLEN))
#     for crange in "-0.577 -0.55" "-0.55 -0.5" "-0.5 0.5" "0.5 0.55" "0.55 0.577"; do # "-0.6 -0.577"; do
#         IFS=' ' read -r -a array <<< "$crange"
#         cmin="${array[0]}"
#         cmax="${array[1]}"
#         tag=c_${cmin}_${cmax}_begin${begin}
#         if [[ "$crange" == "-0.5 0.5" ]]; then npara=`echo {01..25}`; else npara=`echo {01..08}`; fi
#         tags=`for i in $npara; do echo -n ${tag}_${i} " "; done`
#         parallel python generate-SPDE.py --data_size "'2 $end 64 64'" --method che --D 0.01 --tskip 16 --tskip_noise $TSKIP --tbegin $begin --noise 0.04 --cmin $cmin --cmax $cmax  -o ::: $tags
#     done
# done
# #gather-dat.py c*.npy --cat


# else
# for begin in 10000 20000 30000 40000 50000 60000 70000 80000; do
#     end=$((begin+CLIPLEN))
#     for crange in "-0.6 -0.577" "0.577 0.6"; do
#         IFS=' ' read -r -a array <<< "$crange"
#         cmin="${array[0]}"
#         cmax="${array[1]}"
#         tag=c_${cmin}_${cmax}_begin${begin}
#         tags=`for i in {01..50}; do echo -n ${tag}_${i} " "; done`
#         parallel python generate-SPDE.py --data_size "'2 $end 64 64'" --method che --D 0.01 --tskip 16 --tskip_noise $TSKIP --tbegin $begin --noise 0.04 --cmin $cmin --cmax $cmax  -o ::: $tags
#     done
# done
# #gather-dat.py c*.npy --cat

# ### select training structures
# # for i in c*577*.npy; do echo -n $i " "; python -c "import numpy as np; a=np.load('$i'); print(np.max(a)-np.min(a))" ; done  > tmp.txt
# # echo `awk '{if ($2>1.9) p=0.1; else if ($2>1.7) p=0.2; else if ($2>0.7) p=1; else if ($2>0.63) p=0.7; else p=0.05; if( rand()<p) print $1 }' tmp.txt`


# fi

# # with noise test set
# CLIPLEN=8000
# for begin in 2000 30000; do
#     end=$((begin+CLIPLEN))
#     for crange in "-0.577 -0.55" "-0.55 -0.5" "-0.5 0.5" "0.5 0.55" "0.55 0.577"; do # "-0.6 -0.577"; do
#         IFS=' ' read -r -a array <<< "$crange"
#         cmin="${array[0]}"
#         cmax="${array[1]}"
#         tag=c_${cmin}_${cmax}_begin${begin}
#         if [[ "$crange" == "-0.5 0.5" ]]; then npara=`echo {01..04}`; else npara=`echo {01..50}`; fi
#         tags=`for i in $npara; do echo -n ${tag}_${i} " "; done`
#         parallel python generate-SPDE.py --data_size "'1 $end 64 64'" --method che --D 0.01 --tskip 16 --tskip_noise $TSKIP --tbegin $begin --noise 0.04 --cmin $cmin --cmax $cmax  -o ::: $tags
#     done
# done
# gather-dat.py c*.npy --cat --o test.npy
