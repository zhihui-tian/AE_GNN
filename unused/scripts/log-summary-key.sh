#!/bin/zsh

POSITIONAL_ARGS=()
KEY=Valid
COLUMN=7

while [[ $# -gt 0 ]]; do
  case $1 in
    -k|--key)
      KEY="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--column)
      COLUMN="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      f="$1"
      shift # past argument
    #   echo -n "'<grep --text \"$KEY\" \"$1\"|awk \"{print \\\$6}\" ' u 1 w l t \"$f\",";
      # echo $t $t1 $tag  le > /dev/stderr
      # grep --text "$KEY" $f -H  > >(tail -1 |tr '\n' ' '|sed 's:train/.*::') #\
      grep --text "$KEY" $f -H  > >(tail -1 |sed 's:train/.*::') #\
        # > >(sort -g -k $COLUMN|head -n1|awk '{printf("min2 %s %s ",$7,$8)}') \
        # > >(sort -g -k $((COLUMN+1)|head -n1|awk '{printf("min1 %s %s ",$7,$8)}')
      # echo
      ;;
  esac
done
# for i in $*; do
#   grep Valid --text $i -H |;
#   grep Valid --text $i|sort -g -k7|head -n1|awk '{printf("min2 %s %s ",$7,$8)}'
#   grep Valid --text $i|sort -g -k8|head -n1|awk '{printf("min1 %s %s\n",$7,$8)}'
# done

