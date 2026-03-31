__author__ = 'Fei Zhou'

import numpy as np
# from .longclip import longclip
from PIL import Image

class reddot(object):
# class reddot(longclip):
    def load(self, datf, rotate_reddot=True):
        # just load the bmp file
        # datf: semicolon separated list of directories
        # rotate_reddot: if true, rotate red dots, or raw image otherwise

        # from skimage.io import imread_collection
        import glob, os, re
        from align_images import align_images_with_opencv

        regex = re.compile(r"_.*")
        images = []
        for d in datf.split(';'):
            for f in glob.glob(d+'/*.bmp'):
                img = np.array(Image.open(f))
                img = np.stack([img[...,1], img[...,2], img[...,0]-img[...,1]],-1)

                path, fn = os.path.split(f)
                path2 = path.replace('RED', 'without RED')
                tag = fn.replace('.bmp',''); tag=regex.sub('', tag)
                ref = glob.glob(os.path.join(path2, f'*_{tag}.*'))
                if len(ref) <= 0:
                    continue
                ref = ref[0]
                print('debug f ref', f, ref, img.shape)
                ref = np.array(Image.open(ref))

                if rotate_reddot:
                    im2 = align_images_with_opencv(ref, img, keep_channel=True)
                else:
                    ref = align_images_with_opencv(img, ref)
                    im2 = img
                print('debug im2', im2.shape, ref.shape, end='')
                ref = ref.astype('float32')
                images.append(np.stack([(im2[...,2]>230).astype('float32'), (ref-np.mean(ref))/np.std(ref)], 0))
                print(' output', images[-1].shape)
        images = np.array(images)
        return images

    def save(self, dat, path='.'):
        np.save(path+'/data.npy',dat)
        exit()

    def __init__(self, args, datf, train=True, name='', rotate_reddot=True):
        # super().__init__(args, datf, train, name)
        self.configs = args
        self.train = train
        self.dim = args.dim
        data=self.load(datf, rotate_reddot=rotate_reddot)
        self.save(data)
        # if args.data_slice:
        #     f = lambda x: eval(f'x[{args.data_slice}]')
        #     print('data slicing with', args.data_slice)
        #     data = f(data)
        # if args.data_filter:
        #     f = eval(args.data_filter)
        #     # f = eval(f'lambda x: {args.data_filter}')
        #     print('data filtering with', args.data_filter)
        #     print('debug', f(data))
        #     data = data[f(data)]
        # channel_first = args.channel_first
        # if data.ndim == self.dim+2:
        #     # no channel. Let's add a channel
        #     data = data[:,:,None]
        #     channel_first = True
        # if channel_first:
        #     new_ord = list(range(len(data.shape)))
        #     nch = new_ord.pop(-1)
        #     new_ord.insert(-self.dim, nch)
        #     data = data.transpose(new_ord)
        # if args.space_CG:
        #     data_cg = []
        #     frame_shp0 = np.array(data.shape[3:])
        #     frame_shp = np.array(args.frame_shape)
        #     ncg = frame_shp0//frame_shp
        #     assert np.all(ncg*frame_shp == frame_shp0)
        #     shape_cg = list(data.shape[:3]) + list(np.stack([frame_shp, ncg],1).ravel())
        #     print('reshaping spatial shape to', frame_shp)
        #     axis_mean = tuple(range(4,len(shape_cg),2))
        #     shifts = np.array(np.meshgrid(*[np.arange(i) for i in ncg], indexing='ij')).reshape(self.dim,-1).T
        #     data= np.concatenate([np.mean(np.roll(data,shift,axis=tuple(range(3,3+self.dim))).reshape(shape_cg),axis=axis_mean) for shift in shifts])
        # else:
        #     assert np.array_equal(data.shape[3:], args.frame_shape)
        # if args.time_CG > 1:
        #     data_cg = []
        #     shp0 = np.array(data.shape[1:2])
        #     ncg = args.time_CG
        #     shp = shp0//ncg
        #     n0 = ncg*shp[0]
        #     shape_cg = list(data.shape[:1]) + [shp[0], ncg] + list(data.shape[2:])
        #     print('reshaping clip length to', shp[0])
        #     axis_mean = (2,)
        #     shifts = range(shp0[0]-n0+1) #np.array(np.meshgrid(*[np.arange(i) for i in ncg], indexing='ij')).reshape(self.dim,-1).T
        #     data= np.concatenate([np.mean(data[:,shift:shift+n0].reshape(shape_cg),axis=axis_mean) for shift in shifts])
        self.nclip, self.clip_len = data.shape[0], 1
        self.tot_len = args.total_length
        self.frame_step = args.frame_step
        self.tot_len_frame = (self.tot_len-1)*self.frame_step + 1
        start_pos = [np.arange(0,self.clip_len-self.tot_len_frame+1,args.clip_step)+i*self.clip_len for i in range(self.nclip)]
        self.start_pos = np.array(start_pos).ravel()
        self.flat = data.reshape((-1,)+data.shape[2:])

    def shuffle(self):
        np.random.shuffle(self.start_pos)

    def sample(self, nsample):
        nparts = len(self.start_pos)//nsample
        assert nparts>0, ValueError(f'ERROR nsample {nsample} too large, valid start points {len(self.start_pos)}')
        start = np.split(self.start_pos[:nparts*nsample], nparts)
        return np.array([[self.flat[i:i+self.tot_len_frame:self.frame_step] for i in st] for st in start])

    def __getitem__(self, i):
        return np.array(self.flat[i:i+self.tot_len])

    def __len__(self):
        return len(self.start_pos)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=2, help='data dim')
    parser.add_argument('--dat', default='data/', help='data path')
    parser.add_argument('--rotate_raw', action='store_true', help='rotate raw image instead')
    parser.add_argument('--mask', default='', help='if specified, apply mask to --dat')
    args = parser.parse_args()
    if args.mask:
        dat = np.load(args.dat)
        mask = np.array(Image.open(args.mask))
        print('debug mask', np.min(mask), np.max(mask), mask.shape)
        import matplotlib.pyplot as plt
        mask = (mask[...,1]==255) & (mask[...,0]==0) & (mask[...,2]==0)
        dat = dat * (mask.astype('int32')[None,None,...])
        np.save('masked.npy', dat.astype('float32'))
    else:
        dset = reddot(args, args.dat, rotate_reddot=(not args.rotate_raw))
