import shutil
import pickle
import cartopy.crs as ccrs
import glob
from MakeDirs import MakeDirs
import pyproj
import ray
import sys
import logging
import geopandas
import os
import random
import glob
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
smoke_dir = "/scratch1/RDARCH/rda-ghpcs/Rey.Koki/smoke/"
global data_par_dir
data_par_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Mie_SmokeViz/'

def doesnt_already_exists(yr, fn_heads, idx, density):
    for fn_head in fn_heads:
        file_list = glob.glob('{}truth/{}/{}/{}_{}.tif'.format(data_par_dir, yr, density, fn_head, idx))
        if len(file_list) > 0:
            print("FILE THAT ALREADY EXIST:", file_list[0], flush=True)
            return False
        file_list = glob.glob('{}low_iou/{}_{}.tif'.format(data_par_dir, fn_head, idx))
        if len(file_list) > 0:
            print("THIS ANNOTATION FAILED:", file_list[0], flush=True)
            return False
    return True

def mv_files(truth_src):
    yr_dn = dn_dir.split('/')[-2]
    data_src = truth_src.replace('truth','data')
    truth_dst = truth_src.replace('temp_data/{}'.format(yr_dn),'')
    data_dst = data_src.replace('temp_data/{}'.format(yr_dn),'')
    shutil.copyfile(truth_src, truth_dst)
    shutil.copyfile(data_src, data_dst)


def get_lcc_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                    central_latitude=38.5,
                                    standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                    semiminor_axis=6371229))
    return lcc_proj


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


def get_sat_files(best_time, sat_num):

    fs = s3fs.S3FileSystem(anon=True)
    all_fn_heads = []
    all_sat_fns = []

    G18_op_dt = pytz.utc.localize(datetime(2022, 7, 28, 0, 0))
    G17_op_dt = pytz.utc.localize(datetime(2018, 8, 28, 0, 0))
    G17_end_dt = pytz.utc.localize(datetime(2023, 1, 10, 0, 0))
    if sat_num == '17' and best_time >= G18_op_dt:
        sat_num = '18'
    if best_time < G17_op_dt:
        print("G17 not launched yet")
        sat_num = '16'

    hr = best_time.hour
    hr = str(hr).zfill(2)
    yr = best_time.year
    dn = best_time.strftime('%j')

    full_filelist = []
    try:
        full_filelist = fs.ls("noaa-goes{}/ABI-L1b-RadF/{}/{}/{}/".format(sat_num, yr, dn, hr))
    except Exception as e:
        print(e)
    if len(full_filelist) == 0:
        if len(full_filelist) == 0 and sat_num == '18' and best_time < G17_end_dt:
            sat_num = '17'
        try:
            full_filelist = fs.ls("noaa-goes{}/ABI-L1b-RadF/{}/{}/{}/".format(sat_num, yr, dn, hr))
        except Exception as e:
            print("ERROR WITH FS LS")
            print(sat_num, yr, dn, hr)
            print(e)
    if len(full_filelist) > 0:
        sat_fns = get_closest_file(full_filelist, best_time, sat_num)
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



# we need a consistent random shift in the dataset per each annotation
def get_random_xy(size=1024):
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
    centers = smoke.centroid
    smoke_rows = []

    smoke['Start'] = smoke['Start'].apply(smoke_utc)
    smoke['End'] = smoke['End'].apply(smoke_utc)
    sat_fns_to_dl = []

    for idx, row in smoke.iterrows():
        rand_xy = get_random_xy()
        ts_start = smoke.loc[idx]['Start']
        ts_end = smoke.loc[idx]['End']
        row_yr = ts_start.strftime('%Y')
        lat = centers.loc[idx].y
        lon = centers.loc[idx].x
        sat, best_time = sza_best_time(lat, lon, ts_start)

        fn_heads, sat_fns = get_sat_files(best_time, sat)
        if sat_fns:
            if doesnt_already_exists(yr, fn_heads, idx, density):
                for sat_fn_entry in sat_fns:
                    sat_fns_to_dl.extend(sat_fn_entry)
                    smoke_row_ind = {'smoke': smoke, 'idx': idx, 'bounds': bounds.loc[idx], 'density': row['Density'], 'sat_file_locs': [], 'Start': ts_start, 'rand_xy': rand_xy, 'sat_fns': sat_fn_entry}
                    smoke_rows.append(smoke_row_ind)

    sat_fns_to_dl = list(set(sat_fns_to_dl))
    if sat_fns_to_dl:
        ray.init(num_cpus=4)
        ray.get([download_sat_files.remote(sat_file) for sat_file in sat_fns_to_dl])
        ray.shutdown()

    smoke_rows_final = []
    for smoke_row in smoke_rows:
        file_locs = get_file_locations(smoke_row['sat_fns'])
        if len(file_locs) == 3:
            smoke_row['sat_file_locs'] = file_locs
            smoke_rows_final.append(smoke_row)

    return smoke_rows_final

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
        with open('{}smoke_rows_{}{}.pkl'.format(dn_dir, yr, dn), 'wb') as handle:
            pickle.dump(smoke_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(start_dn, end_dn, yr):
    global dn_dir
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    for date in dates:
        dn_dir = '{}temp_data/{}{}/'.format(data_par_dir, date[1], date[0])
        if not os.path.isdir(dn_dir):
            os.mkdir(dn_dir)
            MakeDirs(dn_dir, yr)
        start = time.time()
        print(date)
        iter_smoke(date)
        print("Time elapsed for data download for day {}{}: {}s".format(date[1], date[0], int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
