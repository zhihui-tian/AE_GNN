#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 train|test TSKIP NOISE"
    exit
fi
job=$1
TSKIP=$2
NOISE=$3

# TSKIP=10
# NOISE=0
rm -f c*.npy

if [[ $job == 'train' ]]; then
    CLIP_LEN=20
    begin_list="100 400 800 1200 2000 3000 4000 8000 24000 50000"
else
    CLIP_LEN=400
    begin_list="100 8000 40000"
fi
SIMLEN=$((TSKIP*CLIP_LEN))
#for begin in ${begin_list}; do
for ibegin in `seq 250`; do
    for crange in "-0.995 -0.8" "-0.8 0" "0 0.8" "0.8 0.995"; do
        begin=`python -c 'import numpy as np; print(int(2**(np.random.uniform(6, 15))))'`
        T=`python -c 'import numpy as np; print(np.random.uniform(0.01, 0.25))'`
        end=$((begin+SIMLEN))
        IFS=' ' read -r -a array <<< "$crange"
        cmin="${array[0]}"
        cmax="${array[1]}"
        tag=c_${cmin}_${cmax}_begin${ibegin}
        echo "debug tag ${tag} T ${T}"
        tags=`for i in {00..03}; do echo -n ${tag}_${i} " "; done`
        parallel python generate_SPDE.py --data_size "'1 $end 64 64'" --method general_che --saveT --D=0.001 --dc0=0.2 --tskip $TSKIP --tskip_noise 1 --tbegin $begin --noise $NOISE --cmin $cmin --cmax $cmax --T=$T -o ::: $tags
    done
done
gather-dat.py c*[0-6]?.npy --cat -o train.npy
rm -f c*.npy

# elif [[ $job == 'test' ]]; then 
# CLIP_LEN=600
# SIMLEN=$((TSKIP*CLIP_LEN))
# rm -f c*.npy
# for begin in 1000 6000; do
#     end=$((begin+SIMLEN))
#     for crange in "-0.6 -0.577" "-0.55 0.55" "0.577 0.6"; do
#         IFS=' ' read -r -a array <<< "$crange"
#         cmin="${array[0]}"
#         cmax="${array[1]}"
#         tag=c_${cmin}_${cmax}_begin${begin}
#         tags=`for i in {00..02}; do echo -n ${tag}_${i} " "; done`
#         parallel python generate_SPDE.py --data_size "'1 $end 64 64'" --method che --D 0.01 --tskip $TSKIP --tskip_noise 1 --tbegin $begin --noise $NOISE --cmin $cmin --cmax $cmax  -o ::: $tags
#     done
# done
# gather-dat.py c*.npy --cat -o test.npy
# rm -f c*.npy
# fi
