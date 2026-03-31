from PIL import Image
# import Image
import os
import numpy as np

def crop(infile,height=512,width=512, wstep=0, hstep=0, nrand=0):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    i_choice = list(range(0,imgheight-height,height if hstep==0 else hstep)) + [imgheight-height] + np.random.randint(imgheight-height+1,size=nrand).tolist()
    j_choice = list(range(0,imgwidth-width,  width  if wstep==0 else wstep)) + [imgwidth -width]  + np.random.randint(imgwidth -width+1, size=nrand).tolist()
    # print(f"debug i_choice {i_choice} {j_choice}")
    for i in sorted(set(i_choice)):
        for j in sorted(set(j_choice)):
            box = (j, i, j+width, i+height)
            # print(f'debug {infile} box {box}')
            yield im.crop(box)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--wstep', type=int, default=100)
    parser.add_argument('--hstep', type=int, default=100)
    parser.add_argument('--nrand', type=int, default=0)
    args = parser.parse_args()

    for f in args.files:
        for k,piece in enumerate(crop(f,height=args.height,width=args.width, wstep=args.wstep, hstep=args.hstep, nrand=args.nrand)):
            img=Image.new('RGB', (args.height,args.width), 255)
            img.paste(piece)
            path=f"{f}_{k:04d}.png"
            img.save(path)

