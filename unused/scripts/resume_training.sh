#!/usr/bin/env bash

if [[ $# -eq 0 ]] ; then
    echo "Usage:   $0 \"job-submission-arguments\" jobdir1 jobdir2 ..."
    echo "Example: $0 debug experiment/run1 experiment/run2"
    echo "         $0 \"-q pdebug -W 2:0\" experiment/run1"
    echo "         $0 \"-w ended(jobid) -G msgnn\" experiment/run1"
    exit 0
fi

read -d '' job_script << EOF
#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn msgnn
#BSUB -G amsdnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

device=0
for DIR in __DIR__ ; do
    echo \$DIR \$device
    command=\`grep mode=train \$DIR/config.txt|tail -n1\`
    echo "command is"
    echo \$command
    CUDA_VISIBLE_DEVICES=\$device \$command &>> \$DIR/log &
    device=\$(((device+1)%4))
done
wait
EOF

if [[ "$1" == "debug" ]]; then
    submission_args="-q pdebug -W 2:0"
else
    submission_args="$1"
fi

job_script=${job_script/__DIR__/${@:2}}
# echo "$job_script" | bsub $submission_args -
bsub $submission_args <(echo "$job_script")
# cat <(echo "$job_script")
