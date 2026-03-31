if (( $# < 1 )); then
  echo "$0 --sub subdir_pattern dirs_with_predictions. subdir_pattern=* (default) or e.g. [1-5]0"
  exit 0
fi
if [ "$1" == "--sub" ]; then
  shift; sub=$1; shift
else
  sub='*'
fi

mapfile -t history < <( ls -d1v $@ )
hist1=${history[0]}
nhist=${#history}
nimage=`ls -d1v $hist1/$sub/ |wc -l`
dir1=`ls -d1v $hist1/$sub/ |head -n1`
#echo "debug nhist $nhist nimage $nimage hist1 $hist1 " `ls -d1v $hist1/*/`
steps=`ls -d1v $dir1/pd* |sed 's/.*pd//g;s/\..*//g'`

for step in $steps; do
    images=`for i in ${history[*]}; do ls -d1v $i/$sub/pd$step.*; done`
    # echo "step $step $images"
    montage -mode concatenate -borderwidth 1 -bordercolor red -tile ${nimage}x `ls -d1v $hist1/$sub/gt$step.*` $images  out$step.png
done
convert -delay 20 out*.png -loop 0 true-v-steps.gif
