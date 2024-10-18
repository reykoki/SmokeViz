import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import skimage
from datetime import datetime


def get_data(fn, data_loc):
    data_fn = glob(data_loc + "data/*/*/" + fn)[0]
    truth_fn = glob(data_loc + "truth/*/*/" + fn)[0]
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    truths = skimage.io.imread(truth_fn, plugin='tifffile')
    return RGB, truths

def get_datetime_from_fn(fn):
    start = fn.split('_')[1][1:-1]
    start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    start_readable = start_dt.strftime('%Y/%m/%d %H:%M UTC')
    return start_readable

def get_mesh(num_pixels):
    x = np.linspace(0,num_pixels-1,num_pixels)
    y = np.linspace(0,num_pixels-1,num_pixels)
    X, Y = np.meshgrid(x,y)
    return X,Y

def plot_densities_from_processed_data(fn, data_loc="./sample_data/", close=False, save=False):
    RGB, truths = get_data(fn, data_loc)
    num_pixels = RGB.shape[1]
    X, Y = get_mesh(num_pixels)
    colors = ['red', 'orange', 'yellow']
    plt.figure(figsize=(8, 6),dpi=100)

    plt.imshow(RGB)
    for idx in reversed(range(3)):
        plt.contour(X,Y,truths[:,:,idx],levels =[.99],colors=[colors[idx]])
    plt.tight_layout(pad=0)
    plt.yticks([])
    plt.xticks([])
    plt.title(get_datetime_from_fn(fn), fontsize=18)
    plt.tight_layout(pad=0)#, h_pad=-.5)
    if save:
        plt.savefig('densities.png', dpi=300)
    plt.show()
    if close:
        plt.close()

def plot_RGB(fn, data_loc="./sample_data/"):
    data_fn = glob(data_loc + "data/*/*/" + fn)[0]
    print(get_datetime_from_fn(fn))
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    lat, lon = get_lat_lon(fn, data_loc)
    plt.figure(figsize=(8, 6),dpi=100)
    plt.imshow(RGB)
    plt.yticks(np.linspace(0,255,5), np.round(lat,2), fontsize=12)
    plt.ylabel('latitude (degrees)', fontsize=16)
    plt.xticks(np.linspace(0,255,5), np.round(lon,2), fontsize=12)
    plt.xlabel('longitude (degrees)', fontsize=16)
    plt.title('RGB',fontsize=24)
    plt.tight_layout(pad=0)
    plt.show()

def plot_R_G_B_RGB(fn, data_loc="./sample_data/"):
    data_fn = glob(data_loc + "data/*/*/" + fn)[0]

    fig, ax = plt.subplots(1, 4, figsize=(40,15))
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    print(fn)
    labels = ['Red', 'Green', 'Blue', 'RGB']
    cmaps = ['Reds', 'Greens', 'Blues']
    for idx in range(4):
        if idx < 3:
            #ax[idx].imshow(RGB[:,:,idx], cmap='Greys_r')
            ax[idx].imshow(RGB[:,:,idx], cmap=cmaps[idx])
        else:
            ax[idx].imshow(RGB)
        ax[idx].set_yticks([])
        ax[idx].set_xticks([])
        ax[idx].set_title(labels[idx],fontsize=30)
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = .02)
    plt.tight_layout(pad = 2)
    #plt.margins(0,0)
    plt.show()

def plot_R_G_B(fn, data_loc="./sample_data/"):
    data_fn = glob(data_loc + "data/*/*/" + fn)[0]
    fig, ax = plt.subplots(1, 3, figsize=(15,30))
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    print(fn)
    labels = ['Red Channel', '\"Green\" Channel', 'Blue Channel']
    for idx in range(3):
        ax[idx].imshow(RGB[:,:,idx], cmap='Greys_r')
        ax[idx].set_yticks([])
        ax[idx].set_xticks([])
        ax[idx].set_title(labels[idx],fontsize=20)
    plt.tight_layout(pad=1)
    plt.show()

def plot_labels(fn, data_loc):
    truth_fn = glob(data_loc + "truth/*/*/" + fn)[0]
    fig, ax = plt.subplots(1, 3, figsize=(15,30))
    truths = skimage.io.imread(truth_fn, plugin='tifffile')
    print(fn)
    labels = ['high', 'medium', 'light']
    for den in range(3):
        ax[den].imshow(truths[:,:,den], cmap='Greys_r', vmin=0, vmax=1)
        ax[den].set_yticks([])
        ax[den].set_xticks([])
        ax[den].set_title(labels[den], fontsize=20)
    plt.tight_layout(pad=1)
    plt.show()


def plot_True_Color(fn, data_loc="./sample_data/"):
    data_fn = glob(data_loc + "data/*/*/" + fn)[0]
    #print(get_datetime_from_fn(fn))
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    plt.figure(figsize=(8, 6),dpi=100)
    plt.imshow(RGB)
    plt.xticks([])
    plt.xlabel('longitude (degrees)', fontsize=16)
    plt.title('True Color \n {}'.format(get_datetime_from_fn(fn)),fontsize=18)
    plt.tight_layout(pad=0)
    plt.show()

