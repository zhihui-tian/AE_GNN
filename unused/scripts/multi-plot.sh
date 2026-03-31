#!/bin/bash

use="1:2"
for i in "$@"; do
  case $i in
    -x=*|--xlabel=*)
      xlabel="${i#*=}"
      shift # past argument=value
      ;;
    -y=*|--ylabel=*)
      ylabel="${i#*=}"
      shift # past argument=value
      ;;
    -p=*|--prefix=*)
      prefix="${i#*=}"
      shift # past argument=value
      ;;
    -u=*|--use=*)
      use="${i#*=}"
      shift # past argument=value
      ;;
    # --default)
    #   DEFAULT=YES
    #   shift # past argument with no value
    #   ;;
    *)
      # unknown option
      ;;
  esac
done
# if [[ -n $1 ]]; then
#     echo "Last line of file specified as non-opt/last argument:"
#     tail -1 $1
# fi

for i in $*; do
  echo -n "\"$i\" u ${use} w l t \"$i\"," 
done |sed "s/^/${prefix}\n set xlabel \"${xlabel}\"\nset ylabel \"${ylabel}\"\nplot /;s/,$/\npause 99\n/" | gnuplot