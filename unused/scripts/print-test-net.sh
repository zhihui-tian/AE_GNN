for i in log-CHE-2d.*; do
  tmpf=`mktemp`
  grep RMSE $i |awk '{print $6}' |grep -v 'MAE\|RMSE\|^ *$'> $tmpf
  emin=`sort -g <$tmpf|head -n1`
  emean=`tail -n 10 $tmpf|awk 'BEGIN{x=0;n=0;} {x+=$1;n++;} END{if (n<=0)n=1; print x/n;}'`
  de=`echo $emin $emean|awk '{print $2-$1}'`
  n=`cat $tmpf |wc -l`
  echo $i $n $emin $de
  rm $tmpf
done |sort -g -k3 |awk 'NF>=4'
