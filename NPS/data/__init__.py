__author__ = 'Fei Zhou'

from importlib import import_module
from sklearn.model_selection import train_test_split

def load_data(args, typ, datf, num_workers=1, ds='train'):
    print(f'debug args, typ, datf {args } {typ} {datf}')
    if typ == 'longclip':
        from NPS.data.longclip import longclip
        n_in, n_out, clip_step = (args.n_in,args.n_out,args.clip_step) if ds=='train' else (args.n_in_test,args.n_out_test,args.clip_step_test)
        ds = longclip(args, datf, n_in+n_out, clip_step)
        print(f'Loaded dataset {datf} size {ds.flat.shape} start_pos {ds.start_pos.shape}')
        print(f'start_pos{ds.start_pos}')
        return ds
    # elif typ.startswith('hubbard1band'):
    #     from NPS.data.hubbard1band import load_hubbard1band
    #     # print(f'debug data', args.data, load_hubbard1band(args.data, cache_data=args.cache))
    #     return load_hubbard1band(args.data, cache_data=args.cache, filter=args.datatype[12:])
    # else:
    #     try:
    #         module_name = typ
    #         m = import_module('NPS.data.' + module_name)
    #         return getattr(m, module_name)(args, datf)
    #     except:
    #         idx = typ.rfind('.')
    #         module_name, ds_name = typ[:idx], typ[idx+1:]
    #         m = import_module(typ)
    #         return getattr(m, ds_name)(args, datf)


class Data:
    def __init__(self, args):
        self.statistics = {}
        if args.single_dataset:
            ds = load_data(args, args.datatype, args.data, args.n_threads)
            self.train, self.test = train_test_split(ds, train_size=args.train_split, random_state=args.seed)
            self.statistics = ds.statistics
        else:
            self.train = load_data(args, args.datatype, args.data_train, args.n_threads, 'train') if (args.mode == 'train') else None
            self.test = load_data(args, args.datatype_test, args.data_test, args.n_threads, 'test')
        if args.dataloader:
            if args.dataloader == 'torch':
                from torch.utils.data import DataLoader as loader
            elif args.dataloader == 'geometric':
                from torch_geometric.loader import DataLoader as loader
            else:
                raise ValueError(f'Unknown dataloader {args.dataloader}')
            if self.train is not None:
                self.train = loader(self.train, batch_size=args.batch, shuffle=True)
            self.test = loader(self.test, batch_size=args.batch_test, shuffle=False, drop_last=False)

