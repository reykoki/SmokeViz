import glob
from multiprocessing import Pool
import shutil
import os
import skimage
import numpy as np



global truth_dir
truth_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie2/truth/'


def onehot(a):
    ncols = 4#a.max()+1
    out = np.zeros( (a.size,ncols), dtype=np.uint8 )
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    out = out[:, :, 1:]
    return out[:, :, ::-1]

def get_test_truth():
    s = 10
    heavy = np.zeros((s,s),dtype=np.int8)
    #heavy[3:6, 3:6]=np.ones((3,3),dtype=np.int8)
    med = np.zeros((s,s),dtype=np.int8)
    #med[1:7, 1:7]=np.ones((6,6),dtype=np.int8)
    light = np.ones((s,s),dtype=np.int8)
    light[-1, :]=np.zeros(s,dtype=np.int8)
    truth = np.dstack([heavy, med, light])
    print(light)
    return truth

#truth_fns = glob.glob('/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie2/truth/2024/Heavy/252/*tif')
truth_fns = ['/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie2/truth/2018/Light/101/G16_s20181012230402_35.0_-94.3_361.tif']
truth_fns = ['/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie2/truth/2020/Heavy/242/G17_s20202421400321_40.51_-122.17_113.tif']
for fn in truth_fns:
    truth_img = skimage.io.imread(fn, plugin='tifffile')
    one_hot = skimage.io.imread(fn.replace('truth','truth_one_hot'), plugin='tifffile')
    print(np.sum(truth_img[:,:,0]))
    print(np.sum(truth_img[:,:,1]))
    print(np.sum(truth_img[:,:,2]))
    print(np.sum(one_hot[:,:,0]))
    print(np.sum(one_hot[:,:,1]))
    print(np.sum(one_hot[:,:,2]))
    if (one_hot[:,:,0] == truth_img[:,:,0]).all():
        print('good')
    else:
        print('')
        print("NOT GOOOD")
        print('')

