iGPU=$1
lambda1=$2
for n in 60000 200000; do
  for i in `seq -w 10`; do
    sed "s/epochs [0-9]* /epochs $n /;s/f1 0.01 /f1 $lambda1 /" job-phase > tmp.sh
    CUDA_VISIBLE_DEVICES=$iGPU . tmp.sh
    mv pf_noise.txt pf_noise_${n}_${iGPU}_${i}.txt
  done
done
