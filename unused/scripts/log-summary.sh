
for i in $*; do
  grep Valid --text $i -H |tail -1 |tr '\n' ' '|sed 's:train/.*::';
  grep Valid --text $i|sort -g -k7|head -n1|awk '{printf("min2 %s %s ",$7,$8)}'
  grep Valid --text $i|sort -g -k8|head -n1|awk '{printf("min1 %s %s\n",$7,$8)}'
done

