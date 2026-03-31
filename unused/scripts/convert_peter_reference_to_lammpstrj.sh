f=$1

nA=`grep ' atoms *$' $f |awk '{print $1}'`

cat << EOF
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
$nA
ITEM: BOX BOUNDS pp pp pp
EOF
grep 'hi *$' $f |sed 's/.lo.*//'
echo 'ITEM: ATOMS type x y z'
grep 'Atoms' $f -A999999 |tail -n+3 |awk '{$1=""; print}'