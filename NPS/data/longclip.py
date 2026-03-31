__author__ = 'Fei Zhou'

import numpy as np
from NPS_common.utils import load_array_auto

def register_args(parser):
    pass

def post_process_args(args):
    pass

class longclip:
    def __init__(self, args, datf, tot_len, clip_step, nskip=1):
        self.configs = args
        self.dim = args.dim
        self.nskip = nskip
        data= self.load_data(datf)
        print(f'data shape is{data.shape}')
        # if args.data_slice:
        #     print('data slicing with', args.data_slice)
        #     f = eval(f'lambda x: x[{args.data_slice}]')
        #     data = f(data)
        # if args.data_filter:
        #     print('data filtering with', args.data_filter)
        #     f = eval(f'lambda x: {args.data_filter}')
        #     data = data[f(data)]
        # if args.data_preprocess:
        #     data = self.preprocess(args.data_preprocess, data)
        channel_first = True
        if data.ndim == self.dim+2:
            # no channel. Let's add a channel
            data = data[:,:,None]
        if channel_first:
            new_ord = list(range(len(data.shape)))
            nch = new_ord.pop(-1)
            new_ord.insert(-self.dim, nch)
            data = data.transpose(new_ord)

            
        if args.space_CG:
            data_cg = []
            frame_shp0 = np.array(data.shape[3:])
            frame_shp = np.array(args.frame_shape)
            ncg = frame_shp0//frame_shp
            assert np.all(ncg*frame_shp == frame_shp0)
            shape_cg = list(data.shape[:3]) + list(np.stack([frame_shp, ncg],1).ravel())
            print('reshaping spatial shape to', frame_shp)
            axis_mean = tuple(range(4,len(shape_cg),2))
            shifts = np.array(np.meshgrid(*[np.arange(i) for i in ncg], indexing='ij')).reshape(self.dim,-1).T
            data= np.concatenate([np.mean(np.roll(data,shift,axis=tuple(range(3,3+self.dim))).reshape(shape_cg),axis=axis_mean) for shift in shifts])
        else:
            assert np.array_equal(data.shape[3:], args.frame_shape), ValueError(f'mismatch {data.shape[3:]} {args.frame_shape}')
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
        self.nclip, self.clip_len = data.shape[:2]
        self.tot_len = tot_len
        try:
            self.frame_step = args.frame_step
        except:
            self.frame_step = 1
        self.tot_len_frame = (self.tot_len-1)*self.frame_step + 1
        # clip_step = args.clip_step if train else args.clip_step_test


        # if args.predict_ff: #Start only at the beginning of clips
        #     start_pos = [int(self.clip_len*i) for i in range(self.nclip)]
        #     print('Starting only from beginnings of samples.')
        # else:
        #     start_pos = [np.arange(0,self.clip_len-self.tot_len_frame+1,clip_step)+i*self.clip_len for i in range(self.nclip)]


        self.clp_step=clip_step
        print(self.clip_len)      ### 单个序列长度    75      75
        print(self.tot_len_frame) ### inference多少   2       6
        print(clip_step)          ### 隔多少采样一次   1      10
        print(self.nclip)         ### 多少序列        96      96

        if args.mode == 'train':
            start_pos = [np.arange(0,self.clip_len-self.tot_len_frame+1,clip_step)+i*self.clip_len for i in range(self.nclip)] ### 如果 out大于clip_len,所有都为空，输出一个, original version
        else:
            start_pos = [np.arange(0,self.clip_len-self.tot_len_frame+1,clip_step*3)+i*self.clip_len for i in range(self.nclip)]
        # start_pos = [np.arange(0,self.clip_len-self.clip_len+1,clip_step)+i*self.clip_len for i in range(self.nclip)]   ###不管out是多少，都输出nclip个,容易导致OOM, adapt version

        self.start_pos = np.array(start_pos).ravel()
        if len(self.start_pos) == 0:
             print('longclip(): Warning: Output frames exceed sample length. Is this intentional?')
             self.start_pos=np.array([0])

        print('Starting Indicies: {}, Sample Size: {}'.format(start_pos, self.clip_len))
        if not args.channel_first:
            data = data.transpose(0,1,*tuple(range(3,3+self.dim)),2)

        self.data_shp = data.shape
        self.flat = data.reshape((-1,)+data.shape[2:])

    def load_data(self, f):
        return load_array_auto(f)

    def preprocess(self, pp_opt_str, a):
        print('data preprocessing with', pp_opt_str)
        if 'fft' not in pp_opt_str:
            try:
                from importlib import import_module
                p, m = pp_opt_str.rsplit('.', 1)
                mod = import_module(p)
                f = getattr(mod, m)
            except:
                f = eval(f'lambda x: {pp_opt_str}')
            return f(a)
        import ast
        pp_opt = ast.literal_eval(pp_opt_str)
        import sys
        sys.path.insert(0, '.')
        from NPS_common import smooth
        if pp_opt.get("name", "fft") == "fft":
            return smooth.smooth_array_fft_np(a, keep_frac=(pp_opt['tkeep'],)+((pp_opt['skeep'],)*(self.dim)), nbatch=1, array_only=True)

    def shuffle(self):
        np.random.shuffle(self.start_pos)

    def sample(self, nsample):
        nparts = len(self.start_pos)//nsample
        assert nparts>0, ValueError(f'ERROR nsample {nsample} too large, valid start points {len(self.start_pos)}')
        start = np.split(self.start_pos[:nparts*nsample], nparts)
        return np.array([[self.flat[i:i+self.tot_len_frame:self.frame_step] for i in st] for st in start])

    def __getitem__(self, i):
        j = self.start_pos[i]
        # if self.i_in_out:
        #     return i, self.flat[j:j+self.n_in], self.flat[j+self.n_in:j+self.tot_len]
        return np.array(self.flat[j:j+self.tot_len])

    def __len__(self):
        return len(self.start_pos)

# import random

# class longclip(object):
#   """Class for handling dataset inputs.
#      Each clip has clip_length frames.
#      When providing data, each data point starts from offset, with total_length of which input_length is input.
#      Then the next data point starts from offset+step. By default step = total_length  """

#   def __init__(self, input_param, datf, train=False, name=''):
#     self.datf = datf
#     self.input_data_type = input_param.get('input_data_type', 'float32')
#     self.output_data_type = input_param.get('output_data_type', 'float32')
#     self.minibatch_size = input_param['minibatch_size']
# #    self.is_output_sequence = input_param.get('is_output_sequence', False)
#     self.data = {}
#     self.indices = []
#     self.current_position = 0
#     self.current_batch_size = 0
#     self.current_batch_indices = []
#     self.current_input_length = 0
#     self.current_output_length = 0
#     self.clip_lens = input_param['clip_length']
#     self.dim=input_param['dim']
#     self.tskip = input_param.get('tskip',1)
#     assert self.dim in (2,3), "ERROR expect dim=2,3, got %d"%(self.dim)
#     self.nclip_max = input_param.get('nclip_max', 0)
#     self.load()
#     # the next data point starts from offset+step.
#     self.offset0=input_param.get('offset0', 0)
#     self.offset1=input_param.get('offset1', 0)
#     self.input_length=input_param['input_length']
#     self.total_length=input_param['total_length']
#     self.clip_step= input_param['clip_step']
#     if self.clip_step < 0:
#       self.clip_step = self.total_length
#     self.begin()

#   def load(self):
#     """Load the data."""
#     dat_1 = np.load(self.datf)
#     if isinstance(dat_1, np.ndarray):
#       # npy format
#       self.data = dat_1[:,::self.tskip]
#       if self.data.ndim == 4:
#         assert self.dim == 2, "ERROR 3D clips should have ndim >=5, got 4"
#         self.data = self.data[:,:,:,:,np.newaxis]
#       elif self.data.ndim == 5:
#         if self.dim == 3:
#           self.data = self.data[:,:,:,:,:,np.newaxis]
#         elif self.data.shape[2]<=3: # 1 or 3 channels
#           self.data = self.data.transpose((0,1,3,4,2))
#       elif self.data.ndim == 6:
#         assert self.dim == 3, "ERROR 2D clips should have ndim <=5, got 6"
#         if self.data.shape[2]<=3: # 1 or 3 channels
#           self.data = self.data.transpose((0,1,3,4,5,2))
#       else:
#         raise ValueError("input data should be 4-6 dimensional")
#       self.nclips = len(self.data)
#       if self.nclip_max > 0:
#         self.nclips = min(self.nclips, self.nclip_max)
#         self.data=self.data[:self.nclips]
#       if self.clip_lens != self.data.shape[1]:
#         if self.clip_lens > 0:
#           print("WARNING: clip length mismatch %d %d"%(self.clip_lens, self.data.shape[1]))
#         self.clip_lens = self.data.shape[1]
#       self.data = self.data.reshape((-1,) + self.data.shape[2:])
#     else:
#       # npz format
#       self.data = dat_1['input_raw_data']
#       self.data = self.data.transpose((0,2,3,1))
#       # if variable number of frames per clip, then they should be supplied as .npz['clip_length']
#       try:
#         self.clip_lens = dat['clip_length']
#         self.nclips = len(self.clip_lens)
#       except:
#         self.nclips = len(self.data)//self.clip_lens
#         pass
#     self.dims = self.data.shape[1:]
#     if isinstance(self.clip_lens, int):
#       self.clip_lens = np.full(self.nclips, self.clip_lens)
#     self.clip_istart=np.cumsum(self.clip_lens)
#     self.clip_istart=np.roll(self.clip_istart, 1)
#     self.clip_istart[0]=0
#     print(self.data.shape)

#   def total(self):
#     """Returns the total number of clips."""
#     return len(self.indices)

#   def __len__(self):
#     return self.total()

#   def set_batch(self):
#     if self.current_position + self.minibatch_size <= self.total():
#       self.current_batch_size = self.minibatch_size
#     else:
#       self.current_batch_size = self.total() - self.current_position
#     self.current_batch_indices = self.indices[self.current_position:self
#                                               .current_position +
#                                               self.current_batch_size]
#     self.current_input_length = self.input_length
#     self.current_output_length = self.total_length- self.input_length

#   def begin(self, do_shuffle=True):
#     """Move to the begin of the batch."""
#     self.start_positions = np.random.randint(self.offset0, self.offset1+1, self.nclips)
#     self.indices = np.concatenate([np.arange(self.start_positions[i], self.clip_lens[i]-self.total_length+1, self.clip_step)+self.clip_istart[i] for i in range(self.nclips)])
#     # print('debug long clip beginning', self.indices)
#     if do_shuffle:
#       random.shuffle(self.indices)
#     self.current_position = 0
#     self.set_batch()

#   def next(self):
#     """Move to the next batch."""
#     self.current_position += self.current_batch_size
#     if self.no_batch_left():
#       return None
#     self.set_batch()

#   def no_batch_left(self):
#     if self.current_position > self.total() - self.current_batch_size:
#       return True
#     else:
#       return False

#   def input_batch(self, nframes=None):
#     """Processes for the input batches."""
#     if self.no_batch_left():
#       return None
#     n = self.current_input_length if nframes is None else nframes
#     input_batch = np.zeros((self.current_batch_size, n) +
#                            self.dims, dtype=self.input_data_type)
#     for i in range(self.current_batch_size):
#       batch_ind = self.current_batch_indices[i]
#       begin = batch_ind
#       end = begin + n
#       data_slice = self.data[begin:end]
#       input_batch[i] = data_slice
#       # print('debug input batch', i, n, nframes, input_batch.shape, begin, end, data_slice.shape)
#     input_batch = input_batch.astype(self.input_data_type)
#     return input_batch

#   def output_batch(self):
#     """Processes for the output batches."""
#     if self.no_batch_left():
#       return None
#     raw_dat = self.data
#     if self.is_output_sequence:
#       output_dim = self.dims
#       output_batch = np.zeros((self.current_batch_size,
#                                self.current_output_length) + output_dim, dtype=self.input_data_type)
#     else:
#       output_batch = np.zeros((self.current_batch_size,) +
#                               tuple(self.dims[1]))
#     for i in range(self.current_batch_size):
#       batch_ind = self.current_batch_indices[i]
#       begin = batch_ind
#       end = begin + self.total_length - self.input_length
#       if self.is_output_sequence:
#         data_slice = raw_dat[begin:end]
#         output_batch[i] = data_slice
#       else:
#         data_slice = raw_dat[begin, :, :, :]
#         output_batch[i, :, :, :] = data_slice
#     output_batch = output_batch.astype(self.output_data_type)
#     return output_batch

#   def get_batch(self):
#     return self.input_batch(self.total_length)
# #    input_seq = self.input_batch()
#  #   output_seq = self.output_batch()
#   #  batch = np.concatenate((input_seq, output_seq), axis=1)
#    # return batch
