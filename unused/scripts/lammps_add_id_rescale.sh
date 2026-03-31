RESCALE=1
# cat $1 |awk '{if (NF==9) {print 1, $0;} else print $0}' |sed 's/^1 ITEM/ITEM/;s/ATOMS id/ATOMS type id/' | \
#   awk -v S=$RESCALE '{if (NR>9){$3*=S;$4*=S;$5*=S;} print}' > ${1%.*}.lammps-dump-text
cat $1 |awk '{if (NF==7) {print 1, $0;} else print $0}' |sed 's/^1 ITEM/ITEM/;s/ATOMS id/ATOMS type id/'
