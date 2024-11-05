import pickle
import glob
import ray
import sys
import logging
import geopandas
import os
from datetime import datetime
import numpy as np
import time
import pytz
import shutil
from grab_smoke import get_smoke_fn_url
from create_data import create_data_truth
from pseudo import find_best_data
from get_goes import get_goes_dl_loc

global smoke_dir
smoke_dir = "/scratch1/RDARCH/rda-ghpcs/Rey.Koki/smoke/"
global ray_par_dir
ray_par_dir = "/tmp/"
global pseudo_dir
pseudo_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL_SmokeViz/'
global full_data_dir
full_data_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/full_dataset/'

@ray.remote(max_calls=1)
def iter_rows(smoke_row):
    smoke = smoke_row['smoke']
    idx = smoke_row['idx']
    bounds = smoke_row['bounds']
    density = smoke_row['density']
    rand_xy = smoke_row['rand_xy']
    yr = smoke_row['yr']
    dn = smoke_row['dn']
    file_locs = smoke_row['sat_file_locs']
    if len(file_locs) > 0:
        create_data_truth(file_locs, smoke, idx, yr, dn, density, rand_xy, full_data_dir)

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
    dt = pytz.utc.localize(datetime.strptime(yr+dn, '%Y%j'))
    print('------')
    print(dt)
    print('------')
    fn, url = get_smoke_fn_url(dt)
    smoke_shape_fn = smoke_dir + fn
    if os.path.exists(smoke_shape_fn):
        smoke = geopandas.read_file(smoke_shape_fn)
        with open('{}{}smoke_rows_{}{}.pkl'.format(full_data_dir, 'smoke_rows/', yr, dn), 'rb') as handle:
            smoke_rows = pickle.load(handle)
        #print(smoke_rows)
        if smoke_rows:
            ray_dir = "{}{}{}".format(ray_par_dir,yr,dn)
            if not os.path.isdir(ray_dir):
                os.mkdir(ray_dir)
            ray.init(num_cpus=12, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1', object_store_memory=10**9)
            run_remote(smoke_rows)
            #run_no_ray(smoke_rows)
            ray.shutdown()
            shutil.rmtree(ray_dir)
            find_best_data(yr, dn)
            goes_dl_loc = get_goes_dl_loc(yr, dn)
            goes_files = glob.glob('{}*.nc'.format(goes_dl_loc))
            for goes_fn in goes_files:
                os.remove(goes_fn)

def main(start_dn, end_dn, yr):
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    for date in dates:
        start = time.time()
        print(date)
        iter_smoke(date)
        print("Time elapsed for data creation for day {}{}: {}s".format(date[1], date[0], int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
