#!/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from NPS_common.utils import load_array, str2slice
import pandas as pd
import seaborn as sns
# sns.set_theme(style="ticks")
# sns.set()
matplotlib.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument("datf", type=str, help="data file")
# parser.add_argument("pd", type=str, help="pd data file")
parser.add_argument("-c", default=':', type=str2slice, help="which channel to show, default all ':', e.g. '0:2'")
parser.add_argument("-L", default=None, type=int, help="channel for labels")
parser.add_argument("-o", default='', help="save as image")
options = parser.parse_args()
if options.o: matplotlib.use('Agg')

dat = load_array(options.datf)[:, options.c]
df = pd.DataFrame(dat)
fig = sns.pairplot(df, hue=options.L)

if options.o:
    fig.figure.savefig(options.o)
else:
    plt.show()
