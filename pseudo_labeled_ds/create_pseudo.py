import shutil
import cartopy.crs as ccrs
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from SmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
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
import s3fs
import pytz
import shutil
import wget
from datetime import timedelta
from grab_smoke import get_smoke
from get_sat import get_best_sat


global smoke_dir
smoke_dir = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/new_data/smoke/"
global ray_par_dir
ray_par_dir = "/scratch/alpine/mecr8410/tmp/"
global data_par_dir
data_par_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_2024/'

def get_file_list(idx):
    truth_file_list = []
    truth_file_list = glob.glob('{}truth/*/*/*_{}.tif'.format(dn_dir, idx))
    truth_file_list.sort()
    print(truth_file_list)
    data_file_list = [s.replace('truth','data') for s in truth_file_list]
    print('number of samples for idx:', len(truth_file_list))
    data_dict = {'find': {'truth': truth_file_list, 'data': data_file_list}}
    return data_dict

def compute_iou(preds, truths):
    densities = ['heavy', 'medium', 'low']
    intersection = 0
    union = 0
    for idx, level in enumerate(densities):
        pred = preds[:,idx,:,:]
        true = truths[:,idx,:,:]
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        intersection += (pred + true == 2).sum()
        union += (pred + true >= 1).sum()
    try:
        iou = intersection / union
        return iou
    except Exception as e:
        print(e)
    return 0

def run_model(idx):
    data_dict = get_file_list(idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transforms = transforms.Compose([transforms.ToTensor()])

    test_set = SmokeDataset(data_dict['find'], data_transforms)

    print('there are {} images for this annotation'.format(len(test_set)))

    def get_best_file(dataloader, model, BCE_loss):
        model.eval()
        torch.set_grad_enabled(False)
        # iou has to be more than .01
        best_iou = .01
        ious = []
        best_truth_fn = None
        for idx, data in enumerate(dataloader):
            batch_data, batch_labels, truth_fn = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            preds = model(batch_data)
            iou = compute_iou(preds, batch_labels)
            ious.append(iou)
            if iou > best_iou:
                best_iou = iou
                best_truth_fn = truth_fn
        print("IoU scores: {}".format(ious, flush=True))
        best_iou_idx = ious.index(max(ious))
        print('best IoU index', best_iou_idx)
        if best_truth_fn:
            print('best_truth_fn: ', best_truth_fn)
            return  best_truth_fn[0]

        fn = truth_fn[0].split('/')[-1]
        bad_fn = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/low_iou/{}".format(fn)
        with open(bad_fn, 'w') as fp:
            pass
        return None


    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    model = smp.DeepLabV3Plus(
            encoder_name="timm-efficientnet-b2",
            encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3, # model input channels
            classes=3, # model output channels
    )
    model = model.to(device)

    BCE_loss = nn.BCEWithLogitsLoss()
    chkpt_path = './model/chkpt.pth'
    checkpoint = torch.load(chkpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    best_fn = get_best_file(test_loader, model, BCE_loss)
    return best_fn

def get_indices():
    file_list = glob.glob('{}truth/*/*/*'.format(dn_dir))
    indices = []
    for fn in file_list:
        idx = fn.split('_')[-1].split('.')[0]
        indices.append(idx)
    indices = list(set(indices))
    indices.sort()
    return indices

def mv_files(truth_src, yr_dn, idx):
    coords_src = truth_src.replace('truth','coords')
    data_src = truth_src.replace('truth','data')
    truth_dst = truth_src.replace('temp_data/{}'.format(yr_dn),'')
    coords_dst = coords_src.replace('temp_data/{}'.format(yr_dn),'')
    data_dst = data_src.replace('temp_data/{}'.format(yr_dn),'')
    shutil.copyfile(truth_src, truth_dst)
    shutil.copyfile(coords_src, coords_dst)
    shutil.copyfile(data_src, data_dst)
    if os.path.exists(truth_dst) and os.path.exists(coords_dst) and os.path.exists(data_dst):
        for f in glob.glob(dn_dir+'*/*/*/*_{}.tif'.format(idx)):
            os.remove(f)

def find_best_data(yr, dn):
    yr_dn = yr+dn
    indices = get_indices()
    print(indices)
    for idx in indices:
        best_fn = run_model(idx)
        if best_fn:
            mv_files(best_fn, yr_dn, idx)

def doesnt_already_exists(yr, fn_heads, idx, density):
    final_dest = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/"
    for fn_head in fn_heads:
        file_list = glob.glob('{}truth/{}/{}/{}_{}.tif'.format(final_dest, yr, density, fn_head, idx))
        if len(file_list) > 0:
            print("FILE THAT ALREADY EXIST:", file_list[0], flush=True)
            return False
        file_list = glob.glob('{}low_iou/{}_{}.tif'.format(final_dest, fn_head, idx))
        if len(file_list) > 0:
            print("THIS ANNOTATION FAILED:", file_list[0], flush=True)
            return False
    return True

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
    d = int(size/2)
    A =A[idx[0]-d:idx[0]+d, idx[1]-d:idx[1]+d]
    return A

def get_norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_data(RGB, idx, fn_data, size=256):
    RGB = reshape(RGB, idx, size)
    total = np.sum(RGB)
    if np.sum(total) > 100 and np.sum(total) < 6e5:
        skimage.io.imsave(fn_data, RGB)
        return True
    else:
        return False

def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

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
                              resolution=1000,
                              area_extent=extent)

    new_scn = scn.resample(my_area)
    return new_scn

def create_data_truth(sat_fns, smoke, idx0, yr, density, rand_xy):
    print('idx: ', idx0)
    fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]

    lcc_proj = get_lcc_proj()

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
    scan_start = pytz.utc.localize(scn[composite].attrs['start_time'])
    scan_end = pytz.utc.localize(scn[composite].attrs['end_time'])
    rel_smoke = pick_temporal_smoke(smoke_lcc, scan_start, scan_end)

    # make sure the smoke shape is within the bounds of the
    x = scn[composite].coords['x']
    y = scn[composite].coords['y']
    lon, lat = scn[composite].attrs['area'].get_lonlats()

    corr_data = get_enhanced_image(scn[composite]).data.compute().data
    RGB = np.einsum('ijk->jki', corr_data)
    RGB[np.isnan(RGB)] = 0

    img_shape = scn[composite].shape


    xx = np.tile(x, (len(y),1))
    yy = np.tile(y, (len(x),1)).T

    cent, idx = get_centroid(center, xx, yy, img_shape, rand_xy)

    if cent:
        png_fn_truth = dn_dir + 'temp_png/truth_' + fn_head + '_{}'.format(idx0) + '.png'
        tif_fn_truth = dn_dir + 'truth/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        tif_fn_data = dn_dir + 'data/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        tif_fn_coords = dn_dir + 'coords/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        data_saved = save_data(RGB, idx, tif_fn_data)
        if data_saved:
            truth_saved  = get_truth(x, y, lcc_proj, rel_smoke, idx, png_fn_truth, tif_fn_truth, center, img_shape)
            if truth_saved:
                plot_coords(lat, lon, idx, tif_fn_coords)
    return fn_head


def get_closest_file(fns, best_time, sat_num):
    diff = timedelta(days=100)
    use_fns = []
    for fn in fns:
        starts = []
        if 'C01' in fn:
            s_e = fn.split('_')[3:5]
            start = s_e[0]
            end = s_e[1][0:11]
            C02_fn = 'C02_G{}_{}_{}'.format(sat_num, start, end)
            C03_fn = 'C03_G{}_{}_{}'.format(sat_num, start, end)
            for f in fns:
                if C02_fn in f:
                   C02_fn = f
                elif C03_fn in f:
                   C03_fn = f
            if 'nc' in C02_fn and 'nc' in C03_fn:
                start = s_e[0][1:-3]
                s_dt = pytz.utc.localize(datetime.strptime(start, '%Y%j%H%M'))
                if diff > abs(s_dt - best_time):
                    diff = abs(s_dt - best_time)
                    use_fns = [fn, C02_fn, C03_fn]
    return use_fns


def get_sat_files(smoke_row):
    smoke = smoke_row['smoke']
    idx = smoke_row['idx']
    bounds = smoke_row['bounds']
    density = smoke_row['density']
    row = smoke.loc[idx]

    fs = s3fs.S3FileSystem(anon=True)
    s_dt = row['Start']
    e_dt = row['End']
    tt = s_dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)

    all_fn_heads = []
    all_sat_fns = []
    t = s_dt
    time_list = [t]
    while t < e_dt:
        t += timedelta(minutes=10)
        time_list.append(t)
    sat_num = get_best_sat(s_dt, e_dt, bounds)
    G18_op_dt = pytz.utc.localize(datetime(2022, 7, 28, 0, 0))
    G17_op_dt = pytz.utc.localize(datetime(2018, 8, 28, 0, 0))
    G17_end_dt = pytz.utc.localize(datetime(2023, 1, 10, 0, 0))
    if sat_num == '17' and s_dt >= G18_op_dt:
        sat_num = '18'
    if s_dt < G17_op_dt:
        print("G17 not launched yet")
        sat_num = '16'
    for curr_time in time_list:
        hr = curr_time.hour
        hr = str(hr).zfill(2)
        yr = curr_time.year
        dn = curr_time.strftime('%j')

        full_filelist = []
        view = 'F'
        try:
            full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, view, yr, dn, hr))
        except Exception as e:
            print(e)
        if len(full_filelist) == 0:
            if len(full_filelist) == 0 and sat_num == '18' and curr_time < G17_end_dt:
                sat_num = '17'
            try:
                full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, view, yr, dn, hr))
            except Exception as e:
                print("ERROR WITH FS LS")
                print(sat_num, view, yr, dn, hr)
                print(e)
        if len(full_filelist) > 0:
            sat_fns = get_closest_file(full_filelist, curr_time, sat_num)
            if sat_fns:
                fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
                all_fn_heads.append(fn_head)
                all_sat_fns.append(sat_fns)

    if len(all_sat_fns)>0:
        all_sat_fns = [list(item) for item in set(tuple(row) for row in all_sat_fns)]
        all_fn_heads = list(set(all_fn_heads))
        return all_fn_heads, all_sat_fns
    return None, None

def get_file_locations(use_fns):
    file_locs = []
    goes_dir = dn_dir + 'goes_temp/'
    for file_path in use_fns:
        dl_loc = goes_dir+file_path.split('/')[-1]
        if os.path.exists(dl_loc):
            file_locs.append(dl_loc)
        else:
            print('{} doesnt exist'.format(dl_loc))
    return file_locs

@ray.remote
def download_sat_files(sat_file):
    fs = s3fs.S3FileSystem(anon=True)
    goes_dir = dn_dir + 'goes_temp/'
    fn = sat_file.split('/')[-1]
    dl_loc = goes_dir+fn
    if os.path.exists(dl_loc) is False:
        print('downloading {}'.format(fn))
        fs.get(sat_file, dl_loc)
    return


@ray.remote
def iter_rows(smoke_row):
    smoke = smoke_row['smoke']
    idx = smoke_row['idx']
    bounds = smoke_row['bounds']
    density = smoke_row['density']
    yr = smoke_row['Start'].strftime('%Y')
    rand_xy = smoke_row['rand_xy']
    file_locs = smoke_row['sat_file_locs']
    if len(file_locs) > 0:
        fns = create_data_truth(file_locs, smoke, idx, yr, density, rand_xy)
        return fns

def run_no_ray(smoke_rows):
    for smoke_row in smoke_rows:
        fn_heads = iter_rows(smoke_row)
    return fn_heads

def run_remote(smoke_rows):
    try:
        fn_heads = ray.get([iter_rows.remote(smoke_row) for smoke_row in smoke_rows])
        return fn_heads
    except Exception as e:
        print(e)
        fn_heads = []
        for smoke_row in smoke_rows:
            sat_fns = smoke_row['sat_file_locs']
            fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
            fn_heads.append(fn_head)
        return fn_heads

# we need a consistent random shift in the dataset per each annotation
def get_random_xy(size=256):
    d = int(size/4)
    x_shift = random.randint(int(-1*d), d)
    y_shift = random.randint(int(-1*d), d)
    return (x_shift, y_shift)

def smoke_utc(time_str):
    fmt = '%Y%j %H%M'
    return pytz.utc.localize(datetime.strptime(time_str, fmt))

# create object that contians all the smoke information needed
def create_smoke_rows(smoke):
    fmt = '%Y%j %H%M'
    smoke_fns = []
    bounds = smoke.bounds
    smoke_rows = []
    smoke_lcc = smoke.to_crs(3857)
    smoke_lcc_area = smoke_lcc['geometry'].area

    smoke['Start'] = smoke['Start'].apply(smoke_utc)
    smoke['End'] = smoke['End'].apply(smoke_utc)
    sat_fns_to_dl = []

    for idx, row in smoke.iterrows():
        rand_xy = get_random_xy()
        ts_start = smoke.loc[idx]['Start']
        print(ts_start)
        row_yr = ts_start.strftime('%Y')
        smoke_row = {'smoke': smoke, 'idx': idx, 'bounds': bounds.loc[idx], 'density': row['Density'], 'sat_file_locs': [], 'Start': ts_start, 'rand_xy': rand_xy, 'sat_fns': []}
        fn_heads, sat_fns = get_sat_files(smoke_row)
        if sat_fns:
            if doesnt_already_exists(row_yr, fn_heads, idx, row['Density']):
                for sat_fn_entry in sat_fns:
                    sat_fns_to_dl.extend(sat_fn_entry)
                    smoke_row['sat_fns'] = sat_fn_entry
                    smoke_rows.append(smoke_row)

    sat_fns_to_dl = list(set(sat_fns_to_dl))
    ray.init(num_cpus=12)
    ray.get([download_sat_files.remote(sat_file) for sat_file in sat_fns_to_dl])
    ray.shutdown()

    smoke_rows_final = []
    for smoke_row in smoke_rows:
            file_locs = get_file_locations(smoke_row['sat_fns'])
            if len(file_locs) == 3:
                smoke_row['sat_file_locs'] = file_locs
                smoke_rows_final.append(smoke_row)

    return smoke_rows

# remove large satellite files and the tif files created during corrections
def remove_files(fn_heads):
    fn_heads = list(set(fn_heads))
    print("REMOVING FILES")
    print(fn_heads)
    for head in fn_heads:
        for fn in glob.glob(dn_dir + 'goes_temp/*{}*'.format(head)):
            os.remove(fn)
        s = head.split('s')[1][:13]
        dt = pytz.utc.localize(datetime.strptime(s, '%Y%j%H%M%S'))
        tif_fn = glob.glob('cimss_true_color_sunz_rayleigh_{}{}{}_{}{}{}.tif'.format(dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), dt.strftime('%H'), dt.strftime('%M'), dt.strftime('%S')))
        if tif_fn:
            os.remove(tif_fn[0])

# analysts can only label data that is taken during the daytime, we want to filter for geos data that was within the timeframe the analysts are looking at
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

    if smoke is not None:
        smoke_rows = create_smoke_rows(smoke)
        print(smoke_rows)
        ray_dir = "{}{}{}".format(ray_par_dir,yr,dn)
        if not os.path.isdir(ray_dir):
            os.mkdir(ray_dir)
        ray.init(num_cpus=8, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1')
        fn_heads = run_remote(smoke_rows)
        #fn_heads = run_no_ray(smoke_rows)
        ray.shutdown()
        shutil.rmtree(ray_dir)
        find_best_data(yr, dn)
        if fn_heads:
            remove_files(fn_heads)


def main(start_dn, end_dn, yr):
    global dn_dir
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    for date in dates:
        dn_dir = '{}/temp_data/{}{}/'.format(data_par_dir, date[1], date[0])
        if not os.path.isdir(dn_dir):
            os.mkdir(dn_dir)
            MakeDirs(dn_dir, yr)
        start = time.time()
        print(date)
        iter_smoke(date)
        shutil.rmtree(dn_dir)
        print("Time elapsed for day {}: {}s".format(date, int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
