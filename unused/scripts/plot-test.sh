KEY='mse'

for i in $@; do
  if [ -f $i ]; then
    f=$i
  else
    f=$i
  fi
  echo -n "'<grep --text \"$KEY\" -A50 \"$f\"|tail -n+2 ' u 1 w l t \"$f\",";
done |sed 's/^/set ylabel "'$KEY'"\nset xlabel "t"\nset logscale z; plot /;s/,$/\npause 99\n/' | gnuplot 

