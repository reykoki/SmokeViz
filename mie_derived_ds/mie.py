import shutil
from pyorbital import astronomy
import numpy as np
import pyproj
from datetime import datetime
import pytz
import sys
import os
import glob
import ray
import numpy as np
import time

global ray_par_dir
ray_par_dir = "/tmp/"
global PL_dir
PL_dir = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/Mie/'
global full_data_dir
full_data_dir = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/full_dataset/'

def sample_doesnt_exist(yr, dn, idx, density):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    file_list = glob.glob('{}truth/{}/{}/{}/*_{}_*_{}.tif'.format(Mie_dir, yr, density, dn, sat_num, start_scan, idx))
    if len(file_list) > 0:
        print("FILES THAT ALREADY EXIST:", file_list, flush=True)
        if len(file_list) > 1:
            file_list.sort()
            fns_to_rm = file_list[1:]
            for fn in fns_to_rm:
                os.remove(fn)
                os.remove(fn.replace('truth','data'))
        return False
    bad_file_list = glob.glob('{}low_iou/{}_{}_*_{}.tif'.format(PL_dir, sat_num, start_scan, idx))

    if len(bad_file_list) > 0:
        print("BAD FILES:", bad_file_list, flush=True)
        return False
    return True

def get_file_list(idx, yr, dn):
    truth_file_list = []
    truth_file_list = glob.glob('{}truth/{}/*/{}/*_{}.tif'.format(full_data_dir, yr, dn, idx))
    truth_file_list.sort()
    data_file_list = [s.replace('truth','data') for s in truth_file_list]
    print('number of samples for idx:', len(truth_file_list))
    data_dict = {'find': {'truth': truth_file_list, 'data': data_file_list}}
    return data_dict

def get_dt_from_fn(fn):
    start = fn.split('/')[-1].split('_s')[-1].split('_e')[0][0:13]
    start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    return pytz.utc.localize(start_dt)

def get_center_lat_lon(fn):
    fn_split = fn.split('.tif')[0].split('_')
    lat = fn_split[-3]
    lon = fn_split[-2]
    return float(lat), float(lon)

def get_best_fn_using_sza(truth_fns):
    lat, lon = get_center_lat_lon(truth_fns[0])
    img_times = np.empty(len(truth_fns), dtype=object)
    for i, fn in enumerate(truth_fns):
        img_times[i] = get_dt_from_fn(fn)
    szas = np.zeros(len(img_times))
    for i, t in enumerate(img_times):
        szas[i] = astronomy.sun_zenith_angle(t, lon, lat)
    thresh = 88
    szas[szas>thresh]=1e5 #arbitrarily large number
    best_time_idx = (np.abs(szas - thresh)).argmin()
    best_fn = truth_fns[best_time_idx]
    print(truth_fns)
    print('best_idx:', best_time_idx)
    print('best_fn:', best_fn)
    return best_fn

@ray.remote(max_calls=1)
def run_model(idx_info):
    idx = idx_info['idx']
    yr = idx_info['yr']
    dn = idx_info['dn']
    data_dict = get_file_list(idx, yr, dn)
    truth_fns = data_dict['find']['truth']
    best_fn = get_best_fn_using_sza(truth_fns)
    mv_files(best_fn, yr, dn, idx)
    return

def indices_dont_exist(yr, dn, indices):
    day_list = []
    for idx in indices:
        file_list = glob.glob('{}truth/{}/*/{}/*_{}.tif'.format(PL_dir, yr, dn, idx))
        if len(file_list) > 0:
            print("PSEUDO THAT ALREADY EXIST:", file_list, flush=True)
        low_iou_file_list = glob.glob('{}low_iou/{}{}_{}'.format(PL_dir, yr, dn, idx))
        if len(low_iou_file_list) > 0:
            print("LOW IOU FILE:", low_iou_file_list, flush=True)
        if len(file_list) == 0 and len(low_iou_file_list) == 0:
            day_list.append({'yr': yr, 'dn': dn, 'idx': idx})
    return day_list

def get_indices(yr, dn):
    file_list = glob.glob('{}truth/{}/*/{}/*.tif'.format(full_data_dir, yr, dn))
    indices = []
    fn_heads = []
    for fn in file_list:
        idx = fn.split('_')[-1].split('.')[0]
        indices.append(idx)
    indices = list(set(indices))
    indices.sort()
    day_list = indices_dont_exist(yr, dn, indices)
    return day_list

def mv_files(truth_src, yr, dn, idx):
    truth_dst = truth_src.replace(full_data_dir, PL_dir)
    data_src = truth_src.replace('truth','data')
    data_dst = truth_dst.replace('truth','data')

    #if not os.path.exists(truth_dst):
    #    os.symlink(truth_src, truth_dst)
    #if not os.path.exists(data_dst):
    #    os.symlink(data_src, data_dst)
    #shutil.copyfile(truth_src, truth_dst)
    #shutil.copyfile(data_src, data_dst)
    #if os.path.exists(truth_dst) and os.path.exists(data_dst):
    #    for f in glob.glob(dn_dir+'*/*/*/*_{}.tif'.format(idx)):
    #        os.remove(f)

def find_best_data(yr, dn):
    start = time.time()
    idx_list = get_indices(yr, dn)
    if idx_list:
        print(idx_list)
        ray_dir = "{}{}{}pseudo".format(ray_par_dir,yr,dn)
        if not os.path.isdir(ray_dir):
            os.mkdir(ray_dir)
        ray.init(num_cpus=24, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1', object_store_memory=10**9)
        ray.get([run_model.remote(idx_dict) for idx_dict in idx_list])
        ray.shutdown()
        shutil.rmtree(ray_dir)
    print("Time elapsed for mie data selection for day {}{}: {}s".format(yr, dn, int(time.time() - start)), flush=True)

def main(start_dn, end_dn, yr):
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    #dns.reverse()
    for dn in dns:
        dn = str(dn).zfill(3)
        find_best_data(yr, dn)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
