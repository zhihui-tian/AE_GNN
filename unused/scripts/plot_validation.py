import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+')
parser.add_argument('--key', '-k', type=str, default='validation')
parser.add_argument('--column', '-c', type=int, default=10)
parser.add_argument('--range', '-r', type=str, default='', help='range, e.g. [1e-3:1e-1]')
args = parser.parse_args()

#from glob import glob
if args.range:
    args.range = f'set yrange [{args.range}]; '
cmd = [f'"< grep --text -H {args.key} {f}" u {args.column} w l t "{f}"' for f in args.files]
cmd = f"echo 'set logscale y; {args.range} plot " + ", ".join(cmd) + "; pause 99' | gnuplot"
# print(cmd)
os.system(cmd)
# for i in HST*.dat; do echo -n '"'$i'" w l t "'$i'",'; done |sed 's/^/set xlabel "E_tot"\nset ylabel "HIST"\nplot /;s/,$/\npause 99\n/' | gnuplot