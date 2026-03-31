
for f in *.tpr; do
    base=${f%.tpr};
    dir=${base#md_}
    echo "file $f base $base dir $dir"
    mkdir -p $dir
    python convert_gromacs_traj_to_lammps.py --s $f --f ${base}*.xtc  --o $dir
done
