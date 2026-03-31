#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("weights", help="weights (ker2,bias1,ker1)")
parser.add_argument("val_range", type=float, nargs='+', help="min and max of input values")
#parser.add_argument("--n_in", type=int, default=1, help="number of inputs")
options = parser.parse_args()
wt = np.loadtxt(options.weights).reshape((3,-1))
nfeat = len(wt[0])
vals = np.array(options.val_range).reshape((-1,2))
n_in = len(vals)
assert n_in==1, 'multi dim input NOT implemented yet'

NX=100
x = np.array(np.meshgrid(*(np.linspace(v[0], v[1], NX) for v in vals)))
print('x is', x.shape)
x = x.reshape((len(x), -1)).T
print('x is', x.shape)
y = np.dot(x, wt[2].reshape(n_in, nfeat)) + wt[1]
print('y is', y.shape)
y = 1 / (1 + np.exp(-y)) #y = np.sigmoid(y)
y = np.dot(y, wt[0].reshape(nfeat,1))
print('y is', y.shape)
out_arr = np.hstack([x, y])
np.savetxt('1layer.txt', out_arr)

exit()
xp = tf.placeholder(tf.float64,shape=x.shape)
y = tf.layers.dense(xp, nfeat, activation='sigmoid', 
      kernel_initializer=tf.constant_initializer(wt[1].reshape(n_in, nfeat)),
      bias_initializer=tf.constant_initializer(wt[0]))
y = tf.layers.dense(y, 1, 
      kernel_initializer=tf.constant_initializer(wt[2].reshape(nfeat, 1)))

with tf.Session() as sess:
    pred = sess.run(y, feed_dict={xp:x})
    print('debug prediction', pred)
