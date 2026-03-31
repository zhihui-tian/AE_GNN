# original version
# https://github.com/apacha/CVC-MUSCIMA/blob/master/CompareDatasets.py

#import os
#from typing import Tuple, List

try:
    from cv2 import cv2, countNonZero, cvtColor
except:
    from cv2 import countNonZero, cvtColor
    import cv2

from PIL import Image, ImageChops
#from tqdm import tqdm
#import re
import numpy as np

def align_images_with_opencv(ref, pic, channel=0, mode='Homography', keep_channel=False):
    im_ref = cv2.imread(ref).astype('float32') if isinstance(ref, str) else ref.astype('float32')
    im2 = cv2.imread(pic).astype('float32') if isinstance(pic, str) else pic.astype('float32')
    if im_ref.ndim==3 and im_ref.shape[-1]==3: im_ref=im_ref[...,channel]
    im2_orig = im2[:]
    if im2.ndim==3 and im2.shape[-1]==3:
        im2=im2[...,channel]
        if not keep_channel:
            im2_orig = im2

    # Convert images to grayscale
    #im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

#    im1_gray = cv2.bitwise_not(im1_gray)

    # Find size of image1
    sz = im_ref.shape

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY if mode=='Homography' else cv2.MOTION_AFFINE

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(3 if mode=='Homography' else 2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-7

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im_ref, im2, warp_matrix, warp_mode, criteria,
        inputMask=None, gaussFiltSize=1)
    if mode=='Homography':
        pic_aligned = cv2.warpPerspective(im2_orig, warp_matrix, (sz[1], sz[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        pic_aligned = cv2.warpAffine(im2_orig, warp_matrix, (sz[1], sz[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # print(f'debug ref {ref} pic {pic} out {pic_aligned.__class__}')
    return pic_aligned


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img', default='img.png', help='image to be transformed')
    parser.add_argument('ref', default='ref.png', help='reference')
    parser.add_argument('out', default='out.png', help='output file')
    parser.add_argument('--mode', default='affine', help='Homography or affine')
    parser.add_argument('--keep_channel', action='store_true', help='transform all channels of img')
    parser.add_argument('--channel', type=int, default=0, help='which color channel to use')
    args = parser.parse_args()
    outf= align_images_with_opencv(args.ref, args.img, channel=args.channel, keep_channel=args.keep_channel)
    if args.out[-4:] == '.npy':
        np.save(args.out, np.array(outf))
    else:
        cv2.imwrite(args.out, outf)
    exit()
