for i in $*; do
  grep main.py $i/config*|grep -v test_only |tail -n1|sed 's/main.py/python main.py/' |bsub -q pdebug -W 2:0 ; 
done

