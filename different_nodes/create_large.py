import shutil
import pickle
import cartopy.crs as ccrs
import glob
import matplotlib
from MakeDirs import MakeDirs
import pyproj
import ray
import sys
import logging
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pyresample import create_area_def
import geopandas
from satpy import Scene
from satpy.writers import get_enhanced_image
from PIL import Image, ImageOps
import os
import random
import glob
import skimage
from datetime import datetime
import numpy as np
import time
import pytz
import shutil
from datetime import timedelta
from grab_smoke import *

global smoke_dir
smoke_dir = "/scratch1/RDARCH/rda-ghpcs/Rey.Koki/smoke/"
global ray_par_dir
ray_par_dir = "/tmp/"
global data_par_dir
data_par_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/large_SmokeViz/'

def mv_files(truth_src):
    yr_dn = dn_dir.split('/')[-2]
    data_src = truth_src.replace('truth','data')
    truth_dst = truth_src.replace('temp_data/{}'.format(yr_dn),'')
    data_dst = data_src.replace('temp_data/{}'.format(yr_dn),'')
    shutil.copyfile(truth_src, truth_dst)
    shutil.copyfile(data_src, data_dst)


def pick_temporal_smoke(smoke_shape, t_0, t_f):
    use_idx = []
    bounds = smoke_shape.bounds
    for idx, row in smoke_shape.iterrows():
        end = row['End']
        start = row['Start']
        # the ranges overlap if:
        if t_0-timedelta(minutes=10)<= end and start-timedelta(minutes=10) <= t_f:
            use_idx.append(idx)
    rel_smoke = smoke_shape.loc[use_idx]
    return rel_smoke

def reshape(A, idx, size=1024):
    d = int(size/2)
    A =A[idx[0]-d:idx[0]+d, idx[1]-d:idx[1]+d]
    return A

def save_data(RGB, fn_data, size=1024):
    #RGB = reshape(RGB, idx, size)
    total = np.sum(np.sum(RGB))
    if total > 100 and total < 3e6:
        skimage.io.imsave(fn_data, RGB)
        return True
    else:
        print("TOTAL SUM: ", total)
        return False

def get_rand_center(idx, rand_xy):
    x_o = idx[0] + rand_xy[0]
    y_o = idx[1] + rand_xy[1]
    return (x_o, y_o)

def find_closest_pt(pt_x, pt_y, x, y):
    x_diff = np.abs(x - pt_x)
    y_diff = np.abs(y - pt_y)
    x_diff2 = x_diff**2
    y_diff2 = y_diff**2
    sum_diff = x_diff2 + y_diff2
    dist = sum_diff**(1/2)
    idx = np.unravel_index(dist.argmin(), dist.shape)
    #if distance is less than 1km away
    if np.min(dist) < 1000:
        return idx
    else:
        print("not close enough")
        return None

def get_centroid(center, x, y, img_shape, rand_xy):
    pt_x = center.x
    pt_y = center.y
    idx = find_closest_pt(pt_x, pt_y, x, y)
    if idx:
        rand_idx = get_rand_center(idx, rand_xy)
        return idx, rand_idx
    else:
        return None, None


def plot_truth(x, y, lcc_proj, smoke, png_fn, img_shape):
    fig = plt.figure(figsize=(img_shape[2]/100, img_shape[1]/100), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=lcc_proj)
    smoke.plot(ax=ax, facecolor='black')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(png_fn, dpi=100)
    plt.close(fig)
    img = Image.open(png_fn)
    bw = img.convert('1')
    bw = ImageOps.invert(bw)

    truth = np.asarray(bw).astype('i')
    #truth = reshape(truth, idx)
    #os.remove(png_fn)
    return truth

def get_truth(x, y, lcc_proj, smoke, png_fn, tif_fn, img_shape):

    low_smoke = smoke.loc[smoke['Density'] == 'Light']
    med_smoke = smoke.loc[smoke['Density'] == 'Medium']
    high_smoke = smoke.loc[smoke['Density'] == 'Heavy']

    # high = [1,1,1], med = [0, 1, 1], low = [0, 0, 1]
    low_truth = plot_truth(x, y, lcc_proj, low_smoke, png_fn, img_shape)
    med_truth = plot_truth(x, y, lcc_proj, med_smoke, png_fn, img_shape)
    high_truth = plot_truth(x, y, lcc_proj, high_smoke, png_fn, img_shape)
    low_truth += med_truth + high_truth
    low_truth = np.clip(low_truth, 0, 1)
    med_truth += high_truth
    med_truth = np.clip(med_truth, 0, 1)

    truth_layers = np.dstack([high_truth, med_truth, low_truth])
    if np.sum(truth_layers) > 0:
        skimage.io.imsave(tif_fn, truth_layers)
        return True
    return False

def get_extent(center, rand_xy):
    cent_x = center.x+(rand_xy[0]*2e3) # multipy by 2km resolution
    cent_y = center.y+(rand_xy[1]*2e3)
    x0 = cent_x - 1.024e6
    y0 = cent_y - 1.024e6
    x1 = cent_x + 1.024e6
    y1 = cent_y + 1.024e6
    return [x0, y0, x1, y1]

def get_lcc_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                    central_latitude=38.5,
                                    standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                    semiminor_axis=6371229))
    return lcc_proj

def get_scn(fns, extent):
    scn = Scene(reader='abi_l1b', filenames=fns)

    scn.load(['cimss_true_color_sunz_rayleigh'], generate=False)
    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              projection=get_lcc_proj(),
                              resolution=2000,
                              area_extent=extent)
    new_scn = scn.resample(my_area)
    return new_scn

def create_data_truth(sat_fns, smoke, idx0, yr, density, rand_xy):
    print('idx: ', idx0)
    fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_e2')[0]
    print(fn_head)

    lcc_proj = get_lcc_proj()
    smoke_lcc = smoke.to_crs(lcc_proj)
    centers = smoke_lcc.centroid
    center = centers.loc[idx0]

    try:
        extent = get_extent(center, rand_xy)
    except:
        return fn_head
    try:
        scn = get_scn(sat_fns, extent)
    except Exception as e:
        print(e)
        print('{} did not download, moving on'.format(sat_fns[0]))
        return fn_head

    composite = 'cimss_true_color_sunz_rayleigh'
    scan_start = pytz.utc.localize(scn[composite].attrs['start_time'])
    scan_end = pytz.utc.localize(scn[composite].attrs['end_time'])
    rel_smoke = pick_temporal_smoke(smoke_lcc, scan_start, scan_end)

    # make sure the smoke shape is within the bounds of the
    x = scn[composite].coords['x']
    y = scn[composite].coords['y']
    lon, lat = scn[composite].attrs['area'].get_lonlats()
    mid_pt = int(lon.shape[0]/2)
    lon_cent = np.round(lon[mid_pt, mid_pt], 2)
    lat_cent = np.round(lat[mid_pt, mid_pt], 2)
    fn_head = '{}_{}_{}_{}'.format(fn_head, lat_cent, lon_cent, idx0)

    corr_data = get_enhanced_image(scn[composite]).data.compute().data
    RGB = np.einsum('ijk->jki', corr_data)
    RGB[np.isnan(RGB)] = 0

    img_shape = scn[composite].shape

    png_fn_truth = dn_dir + 'temp_png/truth_' + fn_head + '_{}'.format(idx0) + '.png'
    tif_fn_truth = dn_dir + 'truth/{}/{}/{}.tif'.format(yr, density, fn_head)
    tif_fn_data = dn_dir + 'data/{}/{}/{}.tif'.format(yr, density, fn_head)
    data_saved = save_data(RGB, tif_fn_data)
    if data_saved:
        truth_saved  = get_truth(x, y, lcc_proj, rel_smoke, png_fn_truth, tif_fn_truth, img_shape)
    del scn
    del RGB
    return




@ray.remote(max_calls=1)
def iter_rows(smoke_row):
    smoke = smoke_row['smoke']
    idx = smoke_row['idx']
    bounds = smoke_row['bounds']
    density = smoke_row['density']
    yr = smoke_row['Start'].strftime('%Y')
    rand_xy = smoke_row['rand_xy']
    file_locs = smoke_row['sat_file_locs']
    if len(file_locs) > 0:
        create_data_truth(file_locs, smoke, idx, yr, density, rand_xy)

def run_no_ray(smoke_rows):
    for smoke_row in smoke_rows:
        iter_rows(smoke_row)
    return

def run_remote(smoke_rows):
    try:
        ray.get([iter_rows.remote(smoke_row) for smoke_row in smoke_rows])
        return
    except Exception as e:
        print(e)
        return

def iter_smoke(date):
    dn = date[0]
    yr = date[1]
    s = '{}/{}'.format(yr, dn)
    fmt = '%Y/%j'
    dt = pytz.utc.localize(datetime.strptime(s, fmt))
    month = dt.strftime('%m')
    day = dt.strftime('%d')
    print('------')
    print(dt)
    print('------')
    smoke = get_smoke(dt, smoke_dir)
    fn, url = get_smoke_fn_url(dt)
    smoke_shape_fn = smoke_dir + fn
    if os.path.exists(smoke_shape_fn):
        smoke = geopandas.read_file(smoke_shape_fn)
        with open('{}smoke_rows_{}{}.pkl'.format(dn_dir, yr, dn), 'rb') as handle:
            smoke_rows = pickle.load(handle)
        print(smoke_rows)
        if smoke_rows:
            ray_dir = "{}{}{}".format(ray_par_dir,yr,dn)
            if not os.path.isdir(ray_dir):
                os.mkdir(ray_dir)
            ray.init(num_cpus=24, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1', object_store_memory=10**9)
            run_remote(smoke_rows)
            #run_no_ray(smoke_rows)
            ray.shutdown()
            shutil.rmtree(ray_dir)
            fns = glob.glob('{}truth/*/*/*'.format(dn_dir))
            for fn in fns:
                mv_files(fn)


def main(start_dn, end_dn, yr):
    global dn_dir
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    for date in dates:
        dn_dir = '{}temp_data/{}{}/'.format(data_par_dir, date[1], date[0])
        if os.path.isdir(dn_dir):
            start = time.time()
            print(date)
            iter_smoke(date)
            shutil.rmtree(dn_dir)
            print("Time elapsed for data creation for day {}{}: {}s".format(date[1], date[0], int(time.time() - start)), flush=True)
        else:
            print(dn_dir, 'does not exist, download data first!')

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
