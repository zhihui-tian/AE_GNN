#!/usr/bin/env python
# coding: utf-8

# In[2]:

from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 

sess = tf.compat.v1.InteractiveSession() 

def from_modes(modes): 
    a=np.zeros((N, N)); 
    xy= np.transpose(np.mgrid[0:N,0:N], (1,2,0)) 
    for k, amp, s0 in modes: 
        a+= np.sin( xy.dot(k) + s0 )*amp 
    return a

N=64


# In[3]:


a=from_modes([[2*np.pi/np.array([16, -32]), 1.0, 0.4], [2*np.pi/np.array([64, 16]), 0.9, 2.2]])
#fig, axarr = plt.subplots(1,2)
#axarr[0].imshow(a); axarr[1].imshow(np.pad(a, ((10,10),(10,10)), 'wrap')); plt.show() 
kernel=np.random.normal(0,1,(9,9))  #np.zeros((11,11)); kernel[:,4]=1; kernel[4,:]=1; kernel[4,4]=-16.0;
kernel=kernel[:,:,None,None]
#np.array([[0,1,0],[1, -4, 1],[0,1,0.0]]).reshape((3,3,1,1));



def wrap_pad(input, size, dim=2):
    M1 = tf.concat([input[:,-size:], input, input[:,0:size]], 1)
    if dim==1:
        return M1
    M1 = tf.concat([M1[:,:, -size:], M1, M1[:,:, 0:size]], 2)
    if dim==2:
        return M1
    M1 = tf.concat([M1[:,:,:, -size:], M1, M1[:,:,:, 0:size]], 3)
    if dim==3:
        return M1


def wrap_pad_paddedarray(input, size, dim=2):
    M1 = tf.concat([input[:,-2*size:size], input[:,size:-size], input[:,size:2*size]], 1)
    if dim==1:
        return M1
    M1 = tf.concat([M1[:,:, -2*size:size], M1[:,:,size:-size], M1[:,:, size:2*size]], 2)
    if dim==2:
        return M1
    M1 = tf.concat([M1[:,:,:, -2*size:size], M1[:,:,:,size:-size], M1[:,:,:, size:2*size]], 3)
    if dim==3:
        return M1



#b=tf.nn.conv2d(a[None,:,:,None], kernel, strides=(1,1,1,1), padding='SAME').eval()[0,:,:,0]
#fig, axarr = plt.subplots(1,2)
#axarr[0].imshow(b); axarr[1].imshow(np.pad(b, ((10,10),(10,10)), 'wrap')); plt.show() 


# In[5]:

fig, axarr = plt.subplots(2,7)
val=[]
for npad in range(1,7):
#b=tf.nn.conv2d(np.pad(a, npad, 'wrap')[None,:,:,None], kernel, strides=(1,1,1,1), padding='SAME'#)
#b[:npad]= b[-2*npad:-npad]; b[-npad:]= b[npad:npad*2]
#b[:,:npad]= b[:,-2*npad:-npad]; b[:,-npad:]= b[:, pad:npad*2]
    print('npad', npad)
    b=tf.nn.conv2d(wrap_pad(a[None,:,:,None], npad), kernel, strides=(1,1,1,1), padding='SAME')[:,npad:-npad,npad:-npad]
    b=b.eval()[0,:,:,0]
    print('after periodic conv',b.shape)
    val.append(b)
    #print(b[npad:-npad, npad:-npad].shape)
    axarr[0,npad-1].imshow(b); axarr[1,npad-1].imshow(np.pad(b, ((10,10),(10,10)), 'wrap'))
    # plt.show(); plt.pause(3)
#plt.show()
print([[np.linalg.norm(i-j) for i in val] for j in val])




from src.utils.periodic_convolution import periodic_convolution

conv_p = periodic_convolution(2, True, 9)
conv = periodic_convolution(2, False, 9)
init = tf.constant_initializer(kernel)
#init = tf.random_uniform_initializer(-1,1)
bp = conv_p(a[None,:,:,None], 1, 9, 1, padding='SAME', use_bias=False, kernel_initializer=init)
b = conv(a[None,:,:,None], 1, 9, 1, padding='SAME', use_bias=False, kernel_initializer=init)
tf.global_variables_initializer().run()
bp=bp.eval()[0,:,:,0]
b=b.eval()[0,:,:,0]
axarr[0,6].imshow(np.pad(b, ((10,10),(10,10)), 'wrap'))
axarr[1,6].imshow(np.pad(bp, ((10,10),(10,10)), 'wrap')) 
plt.show()
