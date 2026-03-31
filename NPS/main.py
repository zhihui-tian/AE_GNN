#!/bin/env python
import torch
import os, sys; sys.path.append(os.path.join(sys.path[0], '..'))

from NPS import utility
# from NPS import data
from NPS.data import Data
from NPS import model
from NPS import loss
from NPS.option import args
import time


# print("\n" + "=" * 80)
# print(" FIGURE 3 FINAL ".center(80, "="))
# print("=" * 80 + "\n")

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
if not checkpoint.ok:
    exit()
loader = Data(args)

# for batch in loader.test:
#     print(type(batch))
#     print(batch.shape)
#     print(batch)
#     break 


model = model.Model(args, checkpoint)
if args.print_model:
    print(model)
    try:
        from torchinfo import summary
        summary(model)
    except:
        from NPS.utility import count_parameters
        print(f'Total params: {count_parameters(model)} parameters')
loss = loss.Loss(args, checkpoint) if not args.predict_only else None

# trainer = utility.make_trainer(args, loader, model, loss, checkpoint)  # original train or inference
# trainer = utility.make_trainer(args, loader, model.model, loss, checkpoint) # inference optimize with skip
if args.infer_mode == 'optimize':
    trainer = utility.make_trainer(args, loader, model.model, loss, checkpoint)   ### for inference in latent space
else:
    trainer = utility.make_trainer(args, loader, model, loss, checkpoint)

# # print the allocated gpu after load data
# def memory_diagnostics(verbose=False, reset_stats=True, suffix=''):
#     ndevices=torch.cuda.device_count()
#     for d in range(ndevices):
#         if verbose:
#             dname=torch.cuda.get_device_name(d)
#             dprop=torch.cuda.get_device_properties(d)
#             a=torch.cuda.max_memory_allocated(d)
#             r=torch.cuda.max_memory_reserved(d)
#             t=dprop.total_memory
#             print('Device: {} = {:.3f}/{:.3f} GB. Total = {:.3f} GB {}'.format(dname, a/1e9, r/1e9, t/1e9, suffix))
#         if reset_stats: torch.cuda.reset_peak_memory_stats(device=d)
# memory_diagnostics(verbose=True, suffix='After Dataloader Load')

# job


if args.mode == 'train':
    trainer.train()
    # trainer.test()
elif args.mode == 'predict':
    # model.eval()
    trainer.predict()
    end = time.time()
    print(f"time comparison: {end:.4f} s")
elif args.mode == 'valid':
    # model.eval()
    trainer.validate()
elif args.mode == 'trace':
    from importlib import import_module
    m = import_module(args.export_wrapper)
    wrapper = m.export_wrapper(model, args, data=loader)
    wrapper.export(args.export_file)
    print('*'*100, f'\n written model to {args.export_file}\n','*'*100)
else:
    raise ValueError(f'Unknown job mode {args.mode}')
checkpoint.done()

