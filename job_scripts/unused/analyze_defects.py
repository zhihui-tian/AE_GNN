from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def porosity_density(img, area_size=(25,20), pixel_size=0.17738):
    assert img.ndim == 2
    pad = (np.array(area_size)/pixel_size/2).astype(int)
    import torch
    weight = torch.tensor(np.ones([1,1] + (2*pad+1).tolist(), dtype='int32'))
    img = torch.tensor(img.astype('int32')[None,None,...])
    counts = torch.nn.functional.conv2d(img, weight, padding=pad.tolist())
    # print(f'debug counts {counts.__class__} {counts.shape} {counts}')
    return counts[0,0].numpy()/weight.nelement()

def max_porosity_density(img):
    return np.max(porosity_density(img))

def max_pore_size(img, pixel_size=0.17738):
    import skimage.measure
    label, n = skimage.measure.label(img.astype(int), background=0, return_num=True)
    pore_size = np.bincount(label.ravel())
    # print('debug pores ', pore_size, np.sum(pore_size))
    return (np.max(pore_size[1:]) if len(pore_size)>1 else 0) * (pixel_size**2)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('gt', default='gt.npy', help='GT')
parser.add_argument('pd', default='pd.npy', help='PD')
parser.add_argument('--raw', default='', help='raw image as background')
parser.add_argument('--out', default='', help='save to file')
parser.add_argument('--count_only', action='store_true', help='count the pores and exit')
# parser.add_argument('plot', action='store_true', help='plot raw image superimposed with red dots')
args = parser.parse_args()

gt=np.load(args.gt)[0]
if args.count_only:
    print(f'{args.gt} max_pore_size {max_pore_size(gt):.3f} max_porosity_density% {max_porosity_density(gt)*100:.3f}')
    exit()
pd=np.load(args.pd)[0]
# pd might be binary 0,1, or need to go through sigmoid
if np.mean(np.abs(pd-pd.astype(int))) > 1e-12:
    # not just binary values
    # pd = 1/(1+np.exp(-pd))
    # pd = pd.astype(int)
    pd = (pd>0).float()

cm = confusion_matrix(gt.ravel(),pd.ravel()).astype(float)
(TN, FP, FN, TP) = cm.ravel()
P = TP + FN; N = TN + FP
TPR = TP/P; TNR = TN/N; PPV = TP/(TP+FP); F1 = 2*TP/(2*TP+FP+FN)
# print(cm.ravel().tolist())
print(f'precision TP/(TP+FP) {PPV:.3f} recall TP/(TP+FN) {TPR:.3f} FN/P {FN/P:.3f} FP/P {FP/P:.3f} F1 {F1:.3f} Fowlkes-Mallows {np.sqrt(PPV*TPR):.3f} Matthews {(TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)):.3f}')

if args.raw:
    raw=np.load(args.raw)[0]
    img = ((raw-np.min(raw))/(np.max(raw)-np.min(raw))*255).astype('uint8')[...,None]
    img = np.repeat(img, 3, 2)
    img[gt*pd>0] = np.array([0,255,0])[None,None,:]
    img[gt>pd] = np.array([255,0,0])[None,None,:]
    img[gt<pd] = np.array([255,255,0])[None,None,:]
    if args.out:
        plt.imsave(args.out, img)
    else:
        plt.imshow(img); plt.show()
