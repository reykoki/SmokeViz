import matplotlib
import pyproj
import ray
import sys
import logging
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pyresample import create_area_def
import geopandas
import pandas as pd
from satpy import Scene
from PIL import Image, ImageOps
import os
import random
import glob
import skimage
from datetime import datetime
import numpy as np
import time
import s3fs
import pytz
import shutil
import wget
from suntime import Sun
from datetime import timedelta
from grab_smoke import get_smoke
from build_smoke_object import build_smoke_rows

data_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/scripts/make_data/'

def check_bounds(x, y, bounds):
    if bounds['minx'] > np.min(x) and bounds['maxx'] < np.max(x) and bounds['miny'] > np.min(y) and bounds['maxy'] < np.max(y):
        return True
    else:
        return False

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

def reshape(A, idx, size=256):
    #print('before reshape: ', np.sum(A))
    d = int(size/2)
    A =A[idx[0]-d:idx[0]+d, idx[1]-d:idx[1]+d]
    #print('after reshape: ', np.sum(A))
    return A

def save_data(R, G, B, idx, fn_data, size=256):
    R = reshape(R, idx, size)
    G = reshape(G, idx, size)
    B = reshape(B, idx, size)
    layers = np.dstack([R, G, B])
    total = np.sum(R) + np.sum(G) + np.sum(B)

    if np.sum(total) > 100 and np.sum(total) < 6e5:
        skimage.io.imsave(fn_data, layers)
        return True
    else:
        return False

def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

# for 512x512 images
def get_rand_center(idx, img_shape, size=256):
    d = int(size/4)
    x_o = random.randint(idx[0]-d, idx[0]+d)
    y_o = random.randint(idx[1]-d, idx[1]+d)
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

def get_centroid(center, x, y, img_shape):
    pt_x = center.x
    pt_y = center.y
    idx = find_closest_pt(pt_x, pt_y, x, y)
    if idx:
        rand_idx = get_rand_center(idx, img_shape, size=256)
        return idx, rand_idx
    else:
        return None, None
def plot_coords(lat, lon, idx, tif_fn):
    lat_coords = reshape(lat, idx)
    lon_coords = reshape(lon, idx)
    coords_layers = np.dstack([lat_coords, lon_coords])
    skimage.io.imsave(tif_fn, coords_layers)
    #print(coords_layers)

def plot_truth(x, y, lcc_proj, smoke, png_fn, idx, img_shape):
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
    truth = reshape(truth, idx)
    os.remove(png_fn)
    return truth

def get_truth(x, y, lcc_proj, smoke, idx, png_fn, tif_fn, center, img_shape):

    low_smoke = smoke.loc[smoke['Density'] == 'Light']
    med_smoke = smoke.loc[smoke['Density'] == 'Medium']
    high_smoke = smoke.loc[smoke['Density'] == 'Heavy']

    # high = [1,1,1], med = [0, 1, 1], low = [0, 0, 1]
    low_truth = plot_truth(x, y, lcc_proj, low_smoke, png_fn, idx, img_shape)
    med_truth = plot_truth(x, y, lcc_proj, med_smoke, png_fn, idx, img_shape)
    high_truth = plot_truth(x, y, lcc_proj, high_smoke, png_fn, idx, img_shape)
    low_truth += med_truth + high_truth
    low_truth = np.clip(low_truth, 0, 1)
    med_truth += high_truth
    med_truth = np.clip(med_truth, 0, 1)

    truth_layers = np.dstack([high_truth, med_truth, low_truth])
    if np.sum(truth_layers) > 0:
        skimage.io.imsave(tif_fn, truth_layers)
        return True
    return False

def get_extent(center):
    x0 = center.x - 2.0e5
    y0 = center.y - 2.0e5
    x1 = center.x + 2.0e5
    y1 = center.y + 2.0e5
    return [x0, y0, x1, y1]

def get_scn(fns, extent):
    scn = Scene(reader='abi_l1b', filenames=fns)

    scn.load(['cimss_true_color_sunz_rayleigh'], generate=False)
    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              projection="+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                              resolution=1000,
                              area_extent=extent)

    new_scn = scn.resample(my_area)
    return new_scn

def create_data_truth(sat_fns, smoke, idx0, yr, density):
    print('idx: ', idx0)
    fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]

    lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
    lcc_proj = pyproj.CRS.from_user_input(lcc_str)
    smoke_lcc = smoke.to_crs(lcc_proj)
    centers = smoke_lcc.centroid
    center = centers.loc[idx0]
    try:
        extent = get_extent(center)
    except:
        return fn_head

    try:
        scn = get_scn(sat_fns, extent)
    except:
        try:
            print('data not done downloading!\nwait 30 seconds')
            time.sleep(30)
            scn = get_scn(sat_fns, extent)
        except:
            print('{} wouldnt download, moving on'.format(sat_fns[0]))
            return fn_head

    composite = 'cimss_true_color_sunz_rayleigh'
    lcc_proj = scn[composite].attrs['area'].to_cartopy_crs()
    smoke_lcc = smoke.to_crs(lcc_proj)
    scan_start = pytz.utc.localize(scn[composite].attrs['start_time'])
    scan_end = pytz.utc.localize(scn[composite].attrs['end_time'])
    rel_smoke = pick_temporal_smoke(smoke_lcc, scan_start, scan_end)

    # make sure the smoke shape is within the bounds of the
    x = scn[composite].coords['x']
    y = scn[composite].coords['y']
    lon, lat = scn[composite].attrs['area'].get_lonlats()

    corr_data = scn.save_dataset(composite, compute=False)
    img_shape = scn[composite].shape

    R = corr_data[0][0].compute()
    G = corr_data[0][1].compute()
    B = corr_data[0][2].compute()

    R = normalize(R)
    G = normalize(G)
    B = normalize(B)

    xx = np.tile(x, (len(y),1))
    yy = np.tile(y, (len(x),1)).T

    cent, idx = get_centroid(center, xx, yy, img_shape)

    if cent:
        png_fn_truth = data_dir + 'temp_png/truth_' + fn_head + '_{}'.format(idx0) + '.png'
        tif_fn_truth = data_dir + 'truth/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        tif_fn_data = data_dir + 'data/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        tif_fn_coords = data_dir + 'coords/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        data_saved = save_data(R, G, B, idx, tif_fn_data)

        if data_saved:
            truth_saved  = get_truth(x, y, lcc_proj, rel_smoke, idx, png_fn_truth, tif_fn_truth, center, img_shape)
            if truth_saved:
                plot_coords(lat, lon, idx, tif_fn_coords)
    return fn_head


def iter_rows(smoke_row):
    smoke = smoke_row['smoke']
    idx = smoke_row['idx']
    bounds = smoke_row['bounds']
    density = smoke_row['density']
    yr = smoke_row['Start'].strftime('%Y')
    file_locs = smoke_row['sat_file_locs']

    if len(file_locs) > 0:
        fns = create_data_truth(file_locs, smoke, idx, yr, density)
        return fns
    else:
        print('ERROR NO FILES FOUND FOR best_time: ', best_time)

def run_no_ray(smoke_rows):
    fn_heads = []
    for smoke_row in smoke_rows:
        fn_head = iter_rows(smoke_row)
        fn_heads.append(fn_head)
    return fn_heads

def run_remote(smoke_rows):
    try:
        fn_heads = ray.get([iter_rows.remote(smoke_row) for smoke_row in smoke_rows])
        return fn_heads
    except Exception as e:
        print("ERROR WITH RAY GET")
        print(e)
        print(smoke_rows)
        fn_heads = []
        for smoke_row in smoke_rows:
            sat_fns = smoke_row['sat_file_locs']
            fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
            fn_heads.append(fn_head)
        return fn_heads

# create object that contians all the smoke information needed


# remove large satellite files and the tif files created during corrections
def remove_files(fn_heads):
    fn_heads = list(set(fn_heads))
    truth_fns = []
    print("REMOVING FILES")
    for head in fn_heads:
        for fn in glob.glob(data_dir + 'goes_temp/*{}*'.format(head)):
            os.remove(fn)
        s = head.split('s')[1][:13]
        dt = pytz.utc.localize(datetime.strptime(s, '%Y%j%H%M%S'))
        tif_fn = glob.glob('cimss_true_color_sunz_rayleigh_{}{}{}_{}{}{}.tif'.format(dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), dt.strftime('%H'), dt.strftime('%M'), dt.strftime('%S')))
        if tif_fn:
            os.remove(tif_fn[0])
        truth_fns.extend(glob.glob('{}truth/*/*/{}_*.tif'.format(data_dir, head)))
    truth_fns = list(set(truth_fns))
    return truth_fns

def get_filelist_from_dt(dt):
    yr = dt.strftime("%Y")
    dn = dt.strftime("%j")
    filelist = glob.glob('{}truth/{}/*/*{}{}*.tif'.format(data_dir, yr, yr, dn))
    return filelist

def iter_smoke(date):
    s = '{}{}'.format(date[1], date[0])
    dt = pytz.utc.localize(datetime.strptime(s, '%Y%j'))
    smoke_dir = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/new_data/smoke/"
    smoke = get_smoke(dt, smoke_dir)

    if smoke is not None:
        smoke_rows = build_smoke_rows(smoke, data_dir)
        #ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1')
        #fn_heads = run_remote(smoke_rows)
        fn_heads = run_no_ray(smoke_rows)
        #ray.shutdown()
        if fn_heads:
            fns = remove_files(fn_heads)
        else:
            fns = get_filelist_from_dt(dt)
        fns = [os.path.basename(x) for x in fns]
        print(fns)
        return fns


def main(start_dn, end_dn, yr):
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    print(dns)
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    print(dates)
    for date in dates:
        start = time.time()
        iter_smoke(date)
        print("Time elapsed for day {}: {}s".format(date, int(time.time() - start)), flush=True)


if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
