import os
import ray
from datetime import datetime
import s3fs
import pytz
import shutil
from datetime import timedelta


def get_goes_dl_loc(yr, dn):
    goes_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/GOES/'
    global goes_dl_loc
    goes_dl_loc = '{}{}/{}/'.format(goes_dir, yr, dn)
    return goes_dl_loc

def get_file_locations(use_fns):
    file_locs = []
    for file_path in use_fns:
        fn_dl_loc = goes_dl_loc+file_path.split('/')[-1]
        if os.path.exists(fn_dl_loc):
            file_locs.append(fn_dl_loc)
        else:
            print('{} doesnt exist'.format(fn_dl_loc))
    return file_locs

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


def get_sat_files(time_list, sat_num):

    fs = s3fs.S3FileSystem(anon=True)
    all_fn_heads = []
    all_sat_fns = []

    G18_op_dt = pytz.utc.localize(datetime(2022, 7, 28, 0, 0))
    G17_op_dt = pytz.utc.localize(datetime(2018, 8, 28, 0, 0))
    G17_end_dt = pytz.utc.localize(datetime(2023, 1, 10, 0, 0))
    if sat_num == '17' and time_list[0] >= G18_op_dt:
        sat_num = '18'
    if time_list[0] < G17_op_dt:
        print("G17 not launched yet")
        sat_num = '16'

    for curr_time in time_list:
        hr = curr_time.hour
        hr = str(hr).zfill(2)
        yr = curr_time.year
        dn = curr_time.strftime('%j')

        full_filelist = []
        try:
            full_filelist = fs.ls("noaa-goes{}/ABI-L1b-RadF/{}/{}/{}/".format(sat_num, yr, dn, hr))
        except Exception as e:
            print(e)
        if len(full_filelist) == 0:
            if len(full_filelist) == 0 and sat_num == '18' and curr_time < G17_end_dt:
                sat_num = '17'
            try:
                full_filelist = fs.ls("noaa-goes{}/ABI-L1b-RadF/{}/{}/{}/".format(sat_num, yr, dn, hr))
            except Exception as e:
                print("ERROR WITH FS LS")
                print(sat_num, yr, dn, hr)
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

@ray.remote
def download_sat_files(sat_file):
    fs = s3fs.S3FileSystem(anon=True)
    fn = sat_file.split('/')[-1]
    fn_dl_loc = goes_dl_loc+fn
    print(fn_dl_loc)
    if os.path.exists(fn_dl_loc) is False:
        print('downloading {}'.format(fn))
        fs.get(sat_file, fn_dl_loc)
    return

