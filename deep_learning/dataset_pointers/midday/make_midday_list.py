import pickle
import pytz
from suntime import Sun
from datetime import datetime
from datetime import timedelta
import skimage

with open('pseudo_labeled_ds.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

time_data_dict = {'more_than_2hrs':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                'more_than_3hrs':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                'less_than_2hrs':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                'less_than_3hrs':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}}
                }


def get_dt_from_fn(fn):
    start = fn.split('/')[-1].split('_s')[-1].split('_e')[0][0:13]
    start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    return pytz.utc.localize(start_dt)

def compare_sunrise_sunset_3hrs(sunrise,sunset,dt):
    thresh_3hrs = timedelta(minutes=180)
    if abs(sunrise - dt) < thresh_3hrs or abs(sunset-dt) < thresh_3hrs:
        time_cat = 'less_than_3hrs'
    else:
        time_cat = 'more_than_3hrs'
    return time_cat

def compare_sunrise_sunset_2hrs(sunrise,sunset,dt):
    thresh_2hrs = timedelta(minutes=120)
    if abs(sunrise - dt) < thresh_2hrs or abs(sunset-dt) < thresh_2hrs:
        time_cat = 'less_than_2hrs'
    else:
        time_cat = 'more_than_2hrs'
    return time_cat

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
        time_cat = compare_sunrise_sunset_2hrs(sunset, sunrise, dt)
        time_data_dict[time_cat][split]['truth'].append(truth_fn)
        time_data_dict[time_cat][split]['data'].append(data_fns[idx])
        time_cat = compare_sunrise_sunset_3hrs(sunset, sunrise, dt)
        time_data_dict[time_cat][split]['truth'].append(truth_fn)
        time_data_dict[time_cat][split]['data'].append(data_fns[idx])

sort_fns(data_dict, 'test')
print('original test dataset size: ', len(data_dict['test']['truth']))
print('for testing datasets:')
print('more_than_2hrs: ', len(time_data_dict['more_than_2hrs']['test']['truth']))
print('more_than_3hrs: ', len(time_data_dict['more_than_3hrs']['test']['truth']))
print('less_than_2hrs: ', len(time_data_dict['less_than_2hrs']['test']['truth']))
print('less_than_3hrs: ', len(time_data_dict['less_than_3hrs']['test']['truth']))
#sort_fns(data_dict, 'train')
#sort_fns(data_dict, 'val')

def save_data_dicts(quad):
    with open('{}.pkl'.format(quad), 'wb') as handle:
        pickle.dump(time_data_dict[quad], handle, protocol=pickle.HIGHEST_PROTOCOL)

for time_cat in time_data_dict.keys():
    save_data_dicts(time_cat)
