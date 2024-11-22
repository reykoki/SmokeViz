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
smoke_dir = "$SMOKE_DIR"
global full_data_dir
full_data_dir = "$FULL_DS_DIR"


def doesnt_already_exists(yr, dn, fn_head, idx, density):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    file_list = glob.glob('{}truth/{}/{}/{}/{}_{}_*_{}.tif'.format(full_data_dir, yr, density, dn, sat_num, start_scan, idx))
    if len(file_list) > 0:
        print("FILE THAT ALREADY EXIST:", file_list[0], flush=True)
        return False
    file_list = glob.glob('{}bad_img/{}_{}_*_{}.tif'.format(full_data_dir, yr, density, sat_num, start_scan, idx))
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

# create object that contians all the smoke information needed
def create_smoke_rows(smoke, yr, dn):
    fmt = '%Y%j %H%M'
    smoke_fns = []
    bounds = smoke.bounds
    centers = smoke.centroid
    smoke_rows = []

    smoke['Start'] = smoke['Start'].apply(smoke_utc)
    smoke['End'] = smoke['End'].apply(smoke_utc)
    sat_fns_to_dl = []

    for idx, row in smoke.iterrows():
        #if idx == 89:
        rand_xy = get_random_xy()
        ts_start = smoke.loc[idx]['Start']
        ts_end = smoke.loc[idx]['End']
        lat = centers.loc[idx].y
        lon = centers.loc[idx].x
        sat_num, valid_times = sza_sat_valid_times(lat, lon, ts_start, ts_end)
        density = row['Density']
        if valid_times and sat_num:
            fn_heads, sat_fns = get_sat_files(valid_times, sat_num)
        else:
            sat_fns = None
        if sat_fns:
            for sample_idx, sat_fn_entry in enumerate(sat_fns):
                if doesnt_already_exists(yr, dn, fn_heads[sample_idx], idx, density):
                    sat_fns_to_dl.extend(sat_fn_entry)
                    smoke_row_ind = {'smoke': smoke, 'idx': idx, 'bounds': bounds.loc[idx], 'density': density, 'sat_file_locs': [], 'Start': ts_start, 'rand_xy': rand_xy, 'sat_fns': sat_fn_entry, 'yr': yr, 'dn': dn}
                    smoke_rows.append(smoke_row_ind)

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
        with open('{}{}smoke_rows_{}{}.pkl'.format(full_data_dir, 'smoke_rows/', yr, dn), 'wb') as handle:
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


def get_dates(yr):
    smoke_row_dir = "$SMOKE_ROW_DIR"
    fns = glob.glob('{}smoke_rows_*.pkl'.format(smoke_row_dir))
    sep = ''
    fns = sep.join(fns)
    dn_yr_list = []
    final_dn = 366
    if yr == 2024:
        final_dn = 320
    for dn in range(1, final_dn):
        yrdn = str(yr)+str(dn).zfill(3)
        if yrdn not in fns:
            dn_yr_list.append({'year': yr, 'dn': dn})
    print(dn_yr_list)
    print(len(dn_yr_list))

    return dn_yr_list

if __name__ == '__main__':
    #dn_yr_list = get_dates(yr)
    #for dn_yr in dn_yr_list:
    #    start_dn  = dn_yr['dn']
    #    end_dn  = dn_yr['dn']
    #    yr = dn_yr['year']
    #    main(start_dn, end_dn, yr)
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
