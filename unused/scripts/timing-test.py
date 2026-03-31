import time
import sys

if sys.argv[1] == 'torch':
    import torch
    sys.path.append('../NPS')
    from NPS.model.common import Laplacian_Conv, laplacian_roll

    cuda = torch.device('cuda')
    torch.no_grad()
    # torch.eval()
    lapl_conv = Laplacian_Conv(1,1,3,True)

    shape = torch.Size((16400, 1, 32, 32, 32))
    x = torch.cuda.FloatTensor(shape)
    torch.randn(shape, out=x)

    start = time.time()
    for i in range(1):
        y = lapl_conv(x)
    end = time.time()
    print('laplacian from conv3d',end - start)

    start = time.time()
    for i in range(1):
        y1 = laplacian_roll(x, lvl=2, dim=3, type='pt')
    end = time.time()
    print('laplacian from roll',end - start)

    print('err=', (y-y1).norm(), 'norms of each', y.norm(), y1.norm(), x.norm())

    shape = torch.Size((164, 1, 32, 32, 32))
    x = torch.cuda.FloatTensor(shape)
    torch.randn(shape, out=x)
    conv = torch.nn.Conv3d(1,128,1,bias=False)
    ker = torch.randn(128,1, device=cuda)
    conv.weight= torch.nn.Parameter(ker[:,:,None,None,None])
    start = time.time()
    for i in range(1):
        y = conv(x)
    end = time.time()
    print('fully connected from conv3d(ker1)',end - start)

    lin = torch.nn.Linear(1,128,bias=False)
    lin.weight=torch.nn.Parameter(ker)
    start = time.time()
    for i in range(1):
        y1 = lin(x.transpose_(1,-1)).transpose_(1,-1)
    end = time.time()
    print('fully connected from linear',end - start)
    print('err=', (y-y1).norm(), 'norms of each', y.norm(), y1.norm(), x.norm())

elif sys.argv[1] == 'tensorflow':
    import tensorflow as tf
    tf.enable_eager_execution()
    tfe = tf.contrib.eager
    sys.path.append('../e3d_lstm/src/')
    # import common
    from layers.common import laplacian_conv, laplacian_roll

    # lapl_conv = laplacian_conv(x, dim=3, periodic=True)

    x= tf.random.normal((16400, 32, 32, 32, 1))

    start = time.time()
    for i in range(1):
        y = laplacian_conv(x, dim=3, periodic=True)
    end = time.time()
    print('laplacian from conv3d',end - start)

    start = time.time()
    for i in range(1):
        y1 = laplacian_roll(x, lvl=1, dim=3, type='tf')
    end = time.time()
    print('laplacian from roll',end - start)

    print('err=', tf.norm(y-y1), 'norms of each', tf.norm(y), tf.norm(y1), tf.norm(x))
