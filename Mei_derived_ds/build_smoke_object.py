import s3fs
import glob
from datetime import datetime
from best_time_Mie import get_best_time
from datetime import timedelta
import os
import geopandas
import pytz

def doesnt_already_exists(yr, fn_head, idx, density, data_dir):
    file_list = glob.glob('{}truth/{}/{}/{}_{}.tif'.format(data_dir, yr, density, fn_head, idx))
    if len(file_list) == 0:
        return True
    else:
        print("FILE THAT ALREADY EXIST:", file_list[0], flush=True)
        return False

def get_file_locations(use_fns, data_dir):
    file_locs = []
    fs = s3fs.S3FileSystem(anon=True)
    goes_dir = data_dir + 'goes_temp/'
    for file_path in use_fns:
        fn = file_path.split('/')[-1]
        dl_loc = goes_dir+fn
        file_locs.append(dl_loc)
        if os.path.exists(dl_loc):
            print("{} already exists".format(fn))
        else:
            print('downloading {}'.format(fn))
            fs.get(file_path, dl_loc)
    return file_locs

def get_closest_file(fns, best_time, sat_num):
    diff = timedelta(days=100)
    use_fns = []
    for fn in fns:
        starts = []
        if 'C01' in fn:
            s_e = fn.split('_')[3:5]
            start = s_e[0]
            end = s_e[1]
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
    sat_num, best_time = get_best_time(s_dt, e_dt, bounds)

    if best_time:
        yr = best_time.year
        if sat_num == '17' and yr > 2023:
            sat_num = '18'
        hr = best_time.hour
        hr = str(hr).zfill(2)
        dn = best_time.strftime('%j')
        view = 'C'
        full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, view, yr, dn, hr))
        if len(full_filelist) == 0:
            if yr <= 2018:
                sat_num = '16'
                print("YOU WANTED 17 BUT ITS NOT LAUNCHED")
            elif yr >= 2022:
                sat_num = '18'
            full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, view, yr, dn, hr))
        sat_fns = get_closest_file(full_filelist, best_time, sat_num)
        if sat_fns:
            fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
            return fn_head, sat_fns
    return None, None

def smoke_utc(time_str):
    fmt = '%Y%j %H%M'
    return pytz.utc.localize(datetime.strptime(time_str, fmt))

def build_smoke_rows(smoke, data_dir):
    fmt = '%Y%j %H%M'
    smoke_fns = []
    bounds = smoke.bounds
    smoke_rows = []
    smoke_lcc = smoke.to_crs(3857)
    smoke_lcc_area = smoke_lcc['geometry'].area
    smoke['Start'] = smoke['Start'].apply(smoke_utc)
    smoke['End'] = smoke['End'].apply(smoke_utc)
    for idx, row in smoke.iterrows():
        ts_start = smoke.loc[idx]['Start']
        row_yr = ts_start.strftime('%Y')
        smoke_row = {'smoke': smoke, 'idx': idx, 'bounds': bounds.loc[idx], 'density': row['Density'], 'sat_file_locs': [], 'Start': ts_start}
        fn_head, sat_fns = get_sat_files(smoke_row)
        if sat_fns:
            if doesnt_already_exists(row_yr, fn_head, idx, row['Density'], data_dir):
                file_locs = get_file_locations(sat_fns, data_dir)
                smoke_row['sat_file_locs'] = file_locs
                smoke_rows.append(smoke_row)
    return smoke_rows

