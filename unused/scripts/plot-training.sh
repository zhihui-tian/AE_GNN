#!/bin/bash
# example: $0 -s 'lc 1' -t "no lr stress 2 layer" experiment/forest2d7*nmp2*lr0/log

POSITIONAL_ARGS=()


KEY=Valid
COLUMN=7
tagfile=0
show_label_counter=0
_style_counter=1
MAX_STYLE=9
auto_style=1

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
    --tagfile)
      tagfile=1
      shift # past argument
      ;;
    -s|--style)
      style="$2"
      auto_style=0
      shift
      shift
      ;;
    -t|--tag)
      t1="$2"
      show_label_counter=0
      if (( $auto_style == 1 )); then
        style="lc $_style_counter"
        ((_style_counter++))
        # echo $style $_style_counter $t1 >> /dev/stderr
      fi
      shift
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      f="$1"
      shift # past argument
    #   echo -n "'<grep --text \"$KEY\" \"$1\"|awk \"{print \\\$6}\" ' u 1 w l t \"$f\",";
      if (( $tagfile == 1 )); then
        t="$f"
      elif (( $show_label_counter <= 0 )); then
        t="$t1"
      else
        t=""
      fi
      # echo $t $t1 $tag  le > /dev/stderr
      echo -n "'<grep --text \"$KEY\" \"$f\" ' u $COLUMN w l t \"$t\" $style,";
      ((show_label_counter++))
      ;;
  esac
done |sed 's/^/set ylabel "loss"; set xlabel "iter"; set logscale y; plot /; s/,$/\npause 99\n/' | gnuplot 

