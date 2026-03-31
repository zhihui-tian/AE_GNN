import numpy as np
import os

# x = np.load('/usr/WS2/tian9/KMC_3D_predict/valid.npy')   
# print(x.shape)
# x_val = np.repeat(np.repeat(np.repeat(x, 2, axis=2), 2, axis=3), 2, axis=4)
# print(x_val.shape)
# np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_x2/valid.npy',x_val)


# x = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')
# print(x.shape)
# x_val = np.repeat(np.repeat(np.repeat(x, 3, axis=2), 3, axis=3), 3, axis=4)
# print(x_val.shape)
# np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_96/valid.npy',x_val)


# x = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')
# print(x.shape)
# x_val = np.repeat(np.repeat(np.repeat(x, 4, axis=2), 4, axis=3), 4, axis=4)
# print(x_val.shape)
# np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_128/valid.npy',x_val)

# x = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')
# print(x.shape)
# x_val = np.repeat(np.repeat(np.repeat(x, 5, axis=2), 5, axis=3), 5, axis=4)
# print(x_val.shape)
# np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_160/valid.npy',x_val)




x = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/train.npy')[:3]
print(x.shape)
x_train= np.repeat(np.repeat(np.repeat(x, 2, axis=2), 2, axis=3), 2, axis=4)
print(x_train.shape)
np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_64/train.npy',x_train)
x = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')[:1]
print(x.shape)
x_val = np.repeat(np.repeat(np.repeat(x, 2, axis=2), 2, axis=3), 2, axis=4)
print(x_val.shape)
np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_64/valid.npy',x_val)



x = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/train.npy')[:3]
print(x.shape)
x_train= np.repeat(np.repeat(np.repeat(x, 3, axis=2), 3, axis=3), 3, axis=4)
print(x_train.shape)
np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_96/train.npy',x_train)

x = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')[:1]
print(x.shape)
x_val = np.repeat(np.repeat(np.repeat(x, 3, axis=2), 3, axis=3), 3, axis=4)
print(x_val.shape)
np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_96/valid.npy',x_val)


x = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/train.npy')[:3]
print(x.shape)
x_train= np.repeat(np.repeat(np.repeat(x, 4, axis=2), 4, axis=3), 4, axis=4)
print(x_train.shape)
np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_128/train.npy',x_train)

x = np.load('/usr/WS2/tian9/KMC_3D_2_4_bh_slow/valid.npy')[:1]
print(x.shape)
x_val = np.repeat(np.repeat(np.repeat(x, 4, axis=2), 4, axis=3), 4, axis=4)
print(x_val.shape)
np.save('/usr/WS2/tian9/fake3d_data/KMC_3D_predict_128/valid.npy',x_val)