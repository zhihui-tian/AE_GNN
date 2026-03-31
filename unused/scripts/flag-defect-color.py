#!/bin/env python
import numpy as np
import argparse
from PIL import Image
import os, glob

def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

parser = argparse.ArgumentParser()
parser.add_argument("image", default="file.bmp", help="image with red dots")
parser.add_argument("--min", type=int, help="min pixel val")
parser.add_argument("--max", type=int, help="max pixel val")
options = parser.parse_args()

dir = os.path.dirname(options.image)
arr = np.array(Image.open(options.image))
flag = np.zeros_like(arr[:,:,0], dtype=np.float32)
flag[np.where(arr[...,0]!= arr[...,1])] = 1
print("Found %d red pixels"%(np.sum(flag)))
np.save(dir+'/flag.npy', flag)

raw = glob.glob(dir+'/*.TIFF')[0]
arr = np.array(Image.open(raw))
arr_no_outlier = reject_outliers(arr.ravel())
print(dir," raw min, max", np.min(arr), np.max(arr), "no outlier", 
  np.min(arr_no_outlier), np.max(arr_no_outlier))
arr_rescale = (arr.astype(np.float32)-options.min)/(options.max-options.min)
np.save(dir+'/raw.npy', arr_rescale)


