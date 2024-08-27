import pickle
import numpy as np
import pytz
from suntime import Sun
from datetime import datetime
from datetime import timedelta
import skimage

with open('pseudo_labeled_ds.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

time_data_dict = {'both': [],
                  'sunset': [],
                  'sunrise': []
                 }


def get_dt_from_fn(fn):
    start = fn.split('/')[-1].split('_s')[-1].split('_e')[0][0:13]
    start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    return pytz.utc.localize(start_dt)

def get_sunrise_sunset_dt(lat, lon, dt):
    pos = Sun(lat, lon)
    sunrise = pos.get_sunrise_time(dt)
    sunset = pos.get_sunset_time(dt)
    if sunrise > sunset:
        sunset = pos.get_sunset_time(dt + timedelta(days=1))
    return sunset, sunrise

def sort_fns(ds, split):
    truth_fns = ds[split]['truth']
    data_fns = ds[split]['data']

    for idx, truth_fn in enumerate(truth_fns):
        coords_fn = truth_fn.replace('truth', 'coords')
        coords = skimage.io.imread(coords_fn, plugin='tifffile')
        # middle lat lon
        lat = coords[128][128][0]
        lon = coords[128][128][1]
        dt = get_dt_from_fn(truth_fn)
        sunset, sunrise = get_sunrise_sunset_dt(lat, lon, dt)
        sunset_delta = (sunset-dt).total_seconds()
        sunrise_delta = (dt-sunrise).total_seconds()
        time_data_dict['sunset'].append(sunset_delta)
        time_data_dict['sunrise'].append(sunrise_delta)
        if sunrise_delta < sunset_delta:
            time_data_dict['both'].append(sunrise_delta)
        else:
            time_data_dict['both'].append(sunset_delta)

sort_fns(data_dict, 'test')

with open('time_data_dict.pkl', 'wb') as handle:
    pickle.dump(time_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('sunset: ', len(time_data_dict['sunset']))
print('sunrise: ', len(time_data_dict['sunrise']))
print('average: ', np.sum(time_data_dict['both'])/len(time_data_dict['both']))
