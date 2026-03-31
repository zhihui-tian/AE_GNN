for i in $PWD/results/CHE-2d.i3*/; do /bin/ls -thld $i/*/ |grep -v total|tail -n+2|awk '{print $NF}' |xargs rm -rf; done
