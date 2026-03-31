#!/bin/env python
import numpy as np
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("f_in", help="input array")
    parser.add_argument("shape", help="new shape")
    parser.add_argument('-o', help='output file')
    options = parser.parse_args()
    dat = np.load(options.f_in)
    np.save(options.o, dat.reshape(list(map(int, re.split('\s|,',options.shape)))))

