import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from calendar import monthrange
import calendar
from glob import glob
import pickle
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable

pldr_fp  = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/deep_learning/dataset_pointers/pseudo/pseudo.pkl'

mie_fp = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/deep_learning/dataset_pointers/mie/mie.pkl"

with open(mie_fp, 'rb') as handle:
    mie_data_dict = pickle.load(handle)

with open(pldr_fp, 'rb') as handle:
    pldr_data_dict = pickle.load(handle)
mie_file_list = mie_data_dict['train']['truth']+ mie_data_dict['val']['truth'] + mie_data_dict['test']['truth']
pldr_file_list = pldr_data_dict['train']['truth']+ pldr_data_dict['val']['truth'] + pldr_data_dict['test']['truth']
pldr_file_list.sort()
mie_file_list.sort()
#print(mie_file_list[0:3])
#print(pldr_file_list[0:3])



def parse_info(path):
    """
    Extract year, doy, timestamp, and sample index from filename.
    Example: G16_s20180251545422_30.45_-84.46_376.tif
    """
    m = re.search(r's(\d{4})(\d{3})(\d{2})(\d{2})(\d{2}).*_(\d+)\.tif$', path)
    if not m:
        return None
    year, doy, hh, mm, ss, idx = m.groups()
    return {
        'year': int(year),
        'doy': int(doy),
        'time': datetime.strptime(f"{year} {doy} {hh}{mm}{ss}", "%Y %j %H%M%S"),
        'idx': int(idx),
    }

def match_samples(mie_files, pldr_files):
    # parse both lists
    mie_info = [parse_info(f) for f in mie_files if parse_info(f)]
    pldr_info = [parse_info(f) for f in pldr_files if parse_info(f)]

    matched = []
    unmatched = 0

    for m in mie_info:
        # find any pldr with same year, doy, idx
        match = next((p for p in pldr_info
                      if p['year']==m['year'] and p['doy']==m['doy'] and p['idx']==m['idx']),
                     None)

        if match:
            dt_minutes = abs((match['time'] - m['time']).total_seconds()) / 60
            matched.append(dt_minutes)
        else:
            unmatched += 1


    print("unmatched: ", unmatched)
    #print(matched)

    return matched

# Example data
mie_file_list = mie_file_list[0:3]
pldr_file_list = pldr_file_list[0:3]

time_diff = match_samples(mie_file_list, pldr_file_list)
np.save('time_diff.npy', time_diff)

