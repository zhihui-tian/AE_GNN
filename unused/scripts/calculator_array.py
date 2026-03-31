#!/bin/env python
import numpy as np
import argparse
import os.path
import matplotlib.pyplot as plt
try:
    import seaborn as sn
except:
    pass
# import scipy

parser = argparse.ArgumentParser()
parser.add_argument("command", help="calculator for file, e.g. 'a.txt - 3.1* b.npy [2:]', where the .txt/npy files are matrices. Output saved to npy file")
parser.add_argument("-o", default='output.npy', help="output.npy")

def myload(fn):
    if fn[-4:] == '.npy':
        return np.load(fn)
    else:
        return np.loadtxt(fn)

def calculator_file(cmd):
    # from csld.util.tool import convert_to_matrix
    cmd_translated=" ".join(["myload('%s')"%(x) if os.path.exists(x) else x for x in cmd.split()])
    result= eval(cmd_translated)
    # result= convert_to_matrix(result)
    # print(matrix2text(result))
    if not isinstance(result, np.ndarray):
        result = None
    return result


if __name__ == "__main__":
    options = parser.parse_args()
    out = calculator_file(options.command)
    saver = np.savetxt if options.o[-4:]=='.txt' else np.save
    if out is not None: saver(options.o, out)
