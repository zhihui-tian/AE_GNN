noise=$1
shift

for act in relu sigmoid tanh; do 
    for nhid_n in 16 32 64 128 256 512; do
        for nlayer in 1 2 3 4; do
            nhid=$(python -c "print(','.join(['$nhid_n']*$nlayer))")  
            for lr in 1e-2; do
                echo -n $act noise $noise $nhid $lr " "
                for i in `seq 2`; do
                    python ~/lassen-space/NPS/NPS/scripts/universal_approx.py --act $act --lr $lr --epochs 5000 --nhidden $nhid --noise $noise --data out.npy --dimy 1  |tail -n1 |awk '{print $2}'|sed 's/tensor(//g;s/,//g' |tr '\n' ' ' 
                    # python ~/lassen-space/NPS/NPS/scripts/universal_approx.py --act $act --lr $lr --epochs 5000 --nhidden $nhid --noise $noise |tail -n1 |awk '{print $2}'|sed 's/tensor(//g;s/,//g' |tr '\n' ' '
                done
                echo
            done
        done
    done
done
