# script=scripts/job-LJ20cT.che
# ID=che_CT
# script=scripts/job-LJ20cT.che_Adensity
# ID=che_AT
script=scripts/job-che2cT.che
ID=che_CT

for nhid in 16 32 64 128 256 512 16,16 32,32 64,64 128,128 16,16,16 32,32,32 64,64,64 128,128,128 64,64,64,64; do
for act in relu tanh sigmoid; do
    for lr in 2e-3; do #8e-4 1e-3 2e-3; do
        # for i in `seq 4`; do
           tag="${ID}_n${nhid}_a${act}_lr${lr}"
           sed "s/act=relu/act=${act}/;s/${ID}_/${tag}/;s/lr=1e-3/lr=${lr}/;s/num_hidden=32,32/num_hidden=${nhid}/" $script |bsub -q pdebug -W 2:0
           #-w 'ended(1709233)'
        # done
    done
done
done
