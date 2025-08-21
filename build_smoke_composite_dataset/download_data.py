import shutil
from multiprocessing import Pool
import pickle
import cartopy.crs as ccrs
import glob
import pyproj
import sys
import logging
import geopandas
import os
import random
import glob
from datetime import datetime
import numpy as np
import time
import pytz
import shutil
import wget
from datetime import timedelta
from grab_smoke import get_smoke
from Mie import sza_sat_valid_times
from get_goes import get_sat_files, get_goes_dl_loc, get_file_locations, download_sat_files, check_goes_exists

global smoke_dir
smoke_dir = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/smoke/"
global full_data_dir
full_data_dir = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/full_dataset/"
global composite_data_dir 
composite_data_dir = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/composite_dataset/"

global PL_data_dir 
PL_data_dir = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/PL_2/"


def doesnt_already_exists(yr, dn, fn_head, idx, density):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    file_list = glob.glob('{}truth/{}/{}/{}/{}_{}_*_{}.tif'.format(composite_data_dir, yr, density, dn, sat_num, start_scan, idx))
    if len(file_list) > 0:
        print("FILE THAT ALREADY EXIST:", file_list[0], flush=True)
        return False
    file_list = glob.glob('{}bad_img/{}_{}_*_{}.tif'.format(composite_data_dir, yr, density, sat_num, start_scan, idx))
    if len(file_list) > 0:
        print("THIS IMAGE FAILED:", file_list[0], flush=True)
        return False
    return True

def get_lcc_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                    central_latitude=38.5,
                                    standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                    semiminor_axis=6371229))
    return lcc_proj



# we need a consistent random shift in the dataset per each annotation
def get_random_xy(size=256):
    d = int(size/4)
    x_shift = random.randint(int(-1*d), d)
    y_shift = random.randint(int(-1*d), d)
    return (x_shift, y_shift)

def smoke_utc(time_str):
    fmt = '%Y%j %H%M'
    return pytz.utc.localize(datetime.strptime(time_str, fmt))

def PL_sample_exists(yr, dn, fn_head, idx, density):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    print("HELLO\n\n")
    print('{}truth/{}/{}/{}/{}_{}_*_{}.tif'.format(PL_data_dir, yr, density, dn, sat_num, start_scan, idx))
    file_list = glob.glob('{}truth/{}/{}/{}/{}_{}_*_{}.tif'.format(PL_data_dir, yr, density, dn, sat_num, start_scan, idx))
    if len(file_list) > 0:
        print("PL chose this file:", file_list, flush=True)
        return True 

    bad_file_list = glob.glob('{}low_iou/{}_{}_*_{}.tif'.format(PL_data_dir, sat_num, start_scan, idx))

    if len(bad_file_list) > 0:
        print("LOW IoU FILES:", bad_file_list, flush=True)
        return False
    return False 

def find_PL_smoke_row(smoke_rows):
    checked_smoke_rows = []
    for smoke_row in smoke_rows:
        idx = smoke_row['idx']
        density = smoke_row['density']
        yr = smoke_row['yr']
        dn = smoke_row['dn']
        file_locs = smoke_row['sat_file_locs']
        fn_head = file_locs[0].split('C01_')[-1].split('.')[0].split('_e2')[0]
        PL_sample = PL_sample_exists(yr, dn, fn_head, idx, density)
        if PL_sample:
           return smoke_row 
    return None


# create object that contians all the smoke information needed
def create_smoke_rows(smoke, yr, dn):
    fmt = '%Y%j %H%M'
    smoke_fns = []
    bounds = smoke.bounds
    centers = smoke.centroid
    smoke_rows = []
    print(yr, dn, '\n')

    smoke['Start'] = smoke['Start'].apply(smoke_utc)
    smoke['End'] = smoke['End'].apply(smoke_utc)
    sat_fns_to_dl = []

    for idx, row in smoke.iterrows():
        #if idx == 89:

        print('{}{}smoke_rows_{}{}.pkl'.format(full_data_dir, 'smoke_rows/', yr, dn))
        x = input('stop')
        with open('{}{}smoke_rows_{}{}.pkl'.format(full_data_dir, 'smoke_rows/', yr, dn), 'rb') as handle:
            full_ds_smoke_rows = pickle.load(handle)
        print(full_ds_smoke_rows)
        PL_smoke_row = find_PL_smoke_row(full_ds_smoke_rows)
        sat_fns_to_dl.extend(PL_smoke_row['sat_fns'])
        smoke_rows.append(PL_smoke_row)
        

    sat_fns_to_dl = check_goes_exists(sat_fns_to_dl)

    if sat_fns_to_dl:
        p = Pool(8)
        p.map(download_sat_files, sat_fns_to_dl)

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
    dt = pytz.utc.localize(datetime.strptime("{}{}".format(yr,dn), '%Y%j'))
    print('------')
    print(dt)
    print('------')
    smoke = get_smoke(dt, smoke_dir)
    if smoke is not None:
        goes_dl_loc = get_goes_dl_loc(yr, dn)
        smoke_rows = create_smoke_rows(smoke, yr, dn)
        #print(smoke_rows)
        with open('{}{}smoke_rows_{}{}.pkl'.format(composite_data_dir, 'smoke_rows/', yr, dn), 'wb') as handle:
            pickle.dump(smoke_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(start_dn, end_dn, yr):
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    dates.reverse()
    for date in dates:
        start = time.time()
        print(date)
        iter_smoke(date)
        print("Time elapsed for data download for day {}{}: {}s".format(date[1], date[0], int(time.time() - start)), flush=True)




if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
