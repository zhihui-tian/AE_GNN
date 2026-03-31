#!/bin/env bash

for i in $*; do
  python -c "import numpy as np;a=np.load('$i');np.save('$i',a[...,None])"
done
