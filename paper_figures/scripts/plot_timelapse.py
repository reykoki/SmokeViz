import matplotlib.pyplot as plt
import os
import math

from glob import glob
import numpy as np
import skimage
from datetime import datetime
import cartopy.crs as ccrs
from plot_no_coords import get_mesh, get_data, get_datetime_from_fn


def get_fns(data_dir, idx):
    #fns = [os.path.basename(x) for x in glob(data_dir+'/truth/*/*/*_{}.tif'.format(idx))]
    fns = glob(data_dir+'truth/*/*/*_{}.tif'.format(idx))
    fns = sorted(fns)
    return fns

def timelapse(data_loc, idx_oi):
    print(data_loc)
    fns = get_fns(data_loc, idx_oi)
    n_cols = 5
    n_rows = math.ceil(len(fns)/n_cols)

    if n_rows == 1:
        n_rows = 2
    col_indices = list(range(n_cols)) * n_rows
    row_indices = []
    for i in range(n_rows):
        row_indices.extend([i]*n_cols)
    fig, ax = plt.subplots(n_rows,n_cols, figsize=(n_cols*5, n_rows*5))
    for idx, fn in enumerate(fns):
        row_idx = row_indices[idx]
        col_idx = col_indices[idx]
        #RGB, truths = get_data(fn, data_loc)
        data_fn = fn.replace('truth', 'data')

        RGB = skimage.io.imread(data_fn, plugin='tifffile')
        truths = skimage.io.imread(fn, plugin='tifffile')

        RGB_sum = int(np.sum(RGB))
        num_pixels = RGB.shape[1]
        X, Y = get_mesh(num_pixels)
        colors = ['red', 'orange', 'yellow']
        ax[row_idx, col_idx].imshow(RGB)
        for i in range(3):
            ax[row_idx, col_idx].contour(X,Y,truths[:,:,i],levels =[.99],colors=[colors[i]])
        #ax[row_idx, col_idx].contour(X,Y,truths[:,:,2],levels =[.99],colors=[colors[0]])
        ax[row_idx, col_idx].set_yticks([])
        ax[row_idx, col_idx].set_xticks([])
        ax[row_idx, col_idx].set_title( '{:.2e}'.format(RGB_sum) + ' ' +  get_datetime_from_fn(fn.split('/')[-1]), fontsize=10)
    print(np.sum(truths[:,:,0]))

    plt.tight_layout(pad=2)
    yr_dn = data_loc.split('/')[-2]
    plt.savefig('./timelapse/timelapse_{}_{}.png'.format(yr_dn, idx_oi), dpi=300)
    plt.show()

def get_indices(dn_dir):
    fns = glob(dn_dir+'/truth/*/*/*.tif')
    indices = []
    for fn in fns:
        idx = fns[0].split('_')[-1].split('.tif')[0]
        indices.append(idx)
    indices = list(set(indices))
    return indices


def plot_all(data_dir='/scratch1/RDARCH/rda-ghpcs/Rey.Koki/timelapse/temp_data/'):
    dn_dirs = glob(data_dir+'2021*')
    for dn_dir in dn_dirs:
        dn_dir = dn_dir + '/'
        indices = get_indices(dn_dir)
        for idx in indices:
            timelapse(dn_dir, idx)

#timelapse('/scratch1/RDARCH/rda-ghpcs/Rey.Koki/timelapse/temp_data/2022067/', 34)
timelapse('/scratch1/RDARCH/rda-ghpcs/Rey.Koki/timelapse/temp_data/2021364/', 7)
#plot_all()
