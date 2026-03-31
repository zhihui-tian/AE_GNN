find . -name "fields_*" | parallel -I% --max-args 1  python ~/lassen-space/NPS/scripts/preprocess_2dforest.py % -o %/out.sp.npz
python ~/lassen-space/NPS/scripts/gather-dat.py --cat fields_*/out.sp.npz  -o out.sp.npz
split_dataset.py --validation 0.075 --test 0 out.sp.npz --suffix=.sp.npz

