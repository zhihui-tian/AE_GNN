import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.utils.periodic_convolution import periodic_convolution
conv = periodic_convolution(2, True, 3)
lapl_stencil = np.array([[0,1,0],[1,-4,1],[0,1,0]]).reshape((3,3,1,1)).astype(np.float32)
lapl_ker = tf.constant_initializer(lapl_stencil)
def _lapl(x):
    return conv(x, 1, 3, padding='same', kernel_initializer=lapl_ker, use_bias=False, trainable=False, name='lapl', reuse=tf.AUTO_REUSE)
def lapl_np(x,lvl=0): return (np.roll(x,1,0+lvl)+np.roll(x,-1,0+lvl)+np.roll(x,1,1+lvl)+np.roll(x,-1,1+lvl))-4*x

dat=np.load("/usr/WS2/zhou6/data/CHE-2dfine/train.npy")
#dat= 2*dat-1



ntrain=2
a=tf.placeholder(tf.float32, (ntrain,)+dat[0,0].shape)
aval = tf.layers.dense(tf.layers.dense(a, 64,activation='relu'), 1, use_bias=False)
afunc= (tf.math.square(a)-1)*a
loss = tf.reduce_mean(tf.squared_difference(aval , afunc))
opt = tf.train.AdamOptimizer(learning_rate=3e-3)
#opt = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9)
#opt = tf.train.RMSPropOptimizer(learning_rate=1e-3)
#opt = tf.train.AdagradOptimizer(learning_rate=1e-2)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(400):
        istart = np.random.randint(10,198)
        v1,v2,v3,current_loss, _ = sess.run([a,aval,afunc, loss, train_op], feed_dict={a: dat[0,istart:istart+ntrain]})
        print(i, current_loss)
#plt.plot(v1.ravel(), v2.ravel(), '.', v1.ravel(), v3.ravel(), '*'); plt.show()








Nminibatch=32
def en_functional(c, nfeat=64, polynomial=False):
    if polynomial:
        c2 = tf.math.square(c)
#        freeE = tf.concat([_lapl(c), _lapl(c2), _lapl(c*c2)], -1)
        freeE = tf.concat([c, c2, c*c2], -1)
#        freeE = tf.layers.dense(freeE, 1, use_bias=False)
    else:
        freeE = tf.layers.dense(c, nfeat, activation='relu')
    return tf.layers.dense(freeE, 1, use_bias=False)

def che(x_gen):
    freeE = en_functional(x_gen, nfeat=128, polynomial=False)
    return freeE, _lapl(freeE) + tf.layers.dense(_lapl(_lapl(x_gen)), 1, use_bias=False)
#   features = tf.concat([_lapl(freeE), _lapl(_lapl(x_gen))], -1)
#   return freeE, tf.layers.dense(features, 1, use_bias=False)
x_gen =tf.placeholder(tf.float32, (Nminibatch,)+dat[0,0:2].shape)
freeEn, dxdt = che(x_gen[:,0])
#loss = tf.reduce_mean(tf.math.pow(tf.losses.absolute_difference(x_gen[1:2]-x_gen[0:1], dxdt)*1000,3))
loss = tf.reduce_mean(tf.squared_difference(x_gen[:,1]-x_gen[:,0], dxdt))
opt = tf.train.AdamOptimizer(learning_rate=1e-3)
#opt = tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9)
#opt = tf.train.RMSPropOptimizer(learning_rate=1e-3)
#opt = tf.train.AdagradOptimizer(learning_rate=1e-2)
train_op = opt.minimize(loss)

Nclip=len(dat); Nframe=len(dat[0])
datflat= dat.reshape((-1,)+dat.shape[2:])
clip_indices = [i for i in range(len(datflat)) if (i+1)%Nframe != 0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(80000):
#        istart = np.random.randint(100,Nframe-1)
#        iclip = np.random.randint(Nclip-Nminibatch)
#        traindat = dat[iclip:iclip+Nminibatch,istart:istart+2]
#        traindat = dat[np.random.choice(Nclip, Nminimatch, replace=False)]
        iframe = np.random.choice(clip_indices, Nminibatch, replace=False)
        traindat = datflat[np.vstack([iframe, iframe+1]).T,]
        v1 = traindat[:,0]; v2 = traindat[:,1]
        v3, current_loss, _ = sess.run([dxdt, loss, train_op], feed_dict={x_gen: traindat})
        if i%100==0: print(i, current_loss, np.mean((v1-v2)**2))
    en = sess.run([freeEn], feed_dict={x_gen: traindat})[0]
plt.plot((v2-v1).ravel(), v3.ravel(), '.'); plt.show()
#plt.plot(v1.ravel(), (v2-v1).ravel(), '.', v1.ravel(), v3.ravel(), '*'); plt.show()
plt.plot(v1.ravel(), en.ravel(), '.'); plt.show()

#plt.plot((v2-v1).ravel(), (lapl_np(v1**3 - v1 - lapl_np(v1,1),1)).ravel()*0.025, '.'); plt.show()
#plt.hist((v2-v1).ravel(),bins=100); plt.show()
