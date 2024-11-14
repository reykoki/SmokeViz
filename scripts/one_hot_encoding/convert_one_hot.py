import glob
from multiprocessing import Pool
import shutil
import os
import skimage
import numpy as np



global truth_dir
truth_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie2/truth/'

#truth_fns = glob.glob('/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie2/truth/2024/Heavy/252/*tif')

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

#truth_fns = ['/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie2/truth/2024/Heavy/252/G18_s20242521310213_50.88_-111.93_10.tif']
def create_one_hot(yr):
    print('YEAR:', yr)
    truth_fns = glob.glob('{}{}/*/*/*.tif'.format(truth_dir, yr))
    for fn in truth_fns:
        truth_img = skimage.io.imread(fn, plugin='tifffile')
        numbers = np.sum(truth_img, axis=2)
        k = onehot(numbers)
        #if (k[:,:,0] == truth_img[:,:,0]).all():
        skimage.io.imsave(fn.replace('truth','truth_one_hot'), k)
        #else:
        #    print('SOMETHING WENT WRONGGGG')
        #    print(fn)


yrs = [2018, 2020, 2021, 2022, 2023]
p = Pool(len(yrs))
p.map(create_one_hot, yrs)
