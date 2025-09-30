import os
from datetime import datetime
import pytz
from datetime import timedelta
import boto3
from botocore import UNSIGNED
from botocore.client import Config

global client
client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def get_goes_dl_loc(yr, dn):
    #goes_dir = '$GOES_DIR'
    goes_dir = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/GOES/'
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

def get_mode(dt):
    M3_to_M6 = pytz.utc.localize(datetime(2019, 4, 1, 0, 0)) # April 2019 switch from Mode 3 to Mode 6 (every 15 to 10 mins)
    if dt < M3_to_M6:
        mode = 'M3'
    else:
        mode = 'M6'
    return mode

def get_goes_west(sat_num, dt):
    G18_op_dt = pytz.utc.localize(datetime(2022, 7, 28, 0, 0))
    G17_op_dt = pytz.utc.localize(datetime(2018, 8, 28, 0, 0))
    if dt >= G18_op_dt:
        sat_num = '18'
    if dt < G17_op_dt:
        print("G17 not launched yet")
        sat_num = '16'
    return sat_num

def diagnose_filelist(curr_time, mode, sat_num, yr, dn, hr, mn):
    print('need diagnosis')

    diff = timedelta(minutes=10)
    #C01_prefix = 'ABI-L1b-RadC/{}/{}/{}/OR_ABI-L1b-RadC-{}C01_G{}_s{}{}{}'.format(yr, dn, hr, mode, sat_num, yr, dn, hr)
    #print(C01_prefix)
    #C01_filelist = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C01_prefix)
    #print(C01_filelist)
    use_entry = None
    #if C01_filelist == 0:
    if mode == 'M3':
        mode2 = 'M6'
    else:
        mode2 = 'M3'
    C01_prefix = 'ABI-L1b-RadF/{}/{}/{}/OR_ABI-L1b-RadF-{}C01_G{}_s{}{}{}'.format(yr, dn, hr, mode2, sat_num, yr, dn, hr, mn)
    C01_filelist = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C01_prefix)
    G17_end_dt = pytz.utc.localize(datetime(2023, 1, 10, 0, 0))
    if C01_filelist['KeyCount'] == 0 and sat_num == '18' and curr_time < G17_end_dt:
        sat_num = '17'
        C01_prefix = 'ABI-L1b-RadF/{}/{}/{}/OR_ABI-L1b-RadF-{}C01_G{}_s{}{}{}'.format(yr, dn, hr, mode2, sat_num, yr, dn, hr)
        C01_filelist = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C01_prefix)

    if C01_filelist['KeyCount'] == 0:
        return None, None
    for entry in C01_filelist['Contents']:
        start = entry['Key'].split('_')[3:5][0][1:-3]
        s_dt = pytz.utc.localize(datetime.strptime(start, '%Y%j%H%M'))
        if diff > abs(s_dt - curr_time):
            diff = abs(s_dt - curr_time)
            use_entry = entry['Key']
    return use_entry, C01_prefix

def get_GOES_file_loc(curr_time, mode, sat_num):
    yr = curr_time.year
    dn = curr_time.strftime('%j')
    hr = curr_time.hour
    hr = str(hr).zfill(2)
    mn = curr_time.minute
    C01_prefix = 'ABI-L1b-RadF/{}/{}/{}/OR_ABI-L1b-RadF-{}C01_G{}_s{}{}{}{}'.format(yr, dn, hr, mode, sat_num, yr, dn, hr, mn)
    C01_filelist = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C01_prefix)
    if C01_filelist['KeyCount'] != 1:
        C01_fn, C01_prefix = diagnose_filelist(curr_time, mode, sat_num, yr, dn, hr, mn)
    else:
        C01_fn = C01_filelist['Contents'][0]['Key']
    if C01_fn:
        C02_prefix = C01_prefix.replace('C01', 'C02')
        C03_prefix = C01_prefix.replace('C01', 'C03')
        try:
            C02_fn = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C02_prefix)['Contents'][0]['Key']
            C03_fn = client.list_objects_v2(Bucket='noaa-goes{}'.format(sat_num), Prefix=C03_prefix)['Contents'][0]['Key']
            return [C01_fn, C02_fn, C03_fn]
        except Exception as e:
            print('could not find accomaning files for: {}'.format(C01_fn))
            print(e)
            return []

def get_sat_files(time_list, sat_num):

    all_fn_heads = []
    all_sat_fns = []

    if sat_num == '17':
        sat_num = get_goes_west(sat_num, time_list[0])
    mode = get_mode(time_list[0])

    for curr_time in time_list:
        sat_fns = get_GOES_file_loc(curr_time, mode, sat_num)
        if sat_fns:
            fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
            all_fn_heads.append(fn_head)
            all_sat_fns.append(sat_fns)

    if len(all_sat_fns)>0:
        all_sat_fns = [list(item) for item in set(tuple(row) for row in all_sat_fns)]
        all_fn_heads = list(set(all_fn_heads))
        all_sat_fns.sort() 
        all_fn_heads.sort()
        return all_fn_heads, all_sat_fns
    return None, None

def check_goes_exists(sat_files):
    sat_files = list(set(sat_files))
    sat_fns_to_dl = []
    for sat_file in sat_files:
        fn = sat_file.split('/')[-1]
        fn_dl_loc = goes_dl_loc+fn
        sat_num = fn.split('G')[-1][:2]
        if os.path.exists(fn_dl_loc) is False:
            sat_fns_to_dl.append(sat_file)
    return sat_fns_to_dl

def download_sat_files(sat_file):
    fn = sat_file.split('/')[-1]
    fn_dl_loc = goes_dl_loc+fn
    sat_num = fn.split('G')[-1][:2]
    print('downloading {}'.format(fn))
    try:
        client.download_file(Bucket='noaa-goes{}'.format(sat_num), Key=sat_file, Filename=fn_dl_loc)
    except Exception as e:
        print('boto3 failed')
        print(e)
        pass
    return

