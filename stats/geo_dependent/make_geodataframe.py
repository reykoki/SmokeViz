import pickle
import geopandas
import pandas as pd
import numpy as np
import random
import skimage

with open('filtered.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

geo_data_dict = {'Latitude':[], 'Longitude': []}


def sort_fns(ds, split):
    truth_fns = ds[split]['truth']
    for idx, truth_fn in enumerate(truth_fns):
        try:
            density = truth_fn.split('/')[-2]
            yr = truth_fn.split('/')[-3]
            coords_fn = truth_fn.replace('truth', 'coords')
            coords = skimage.io.imread(coords_fn, plugin='tifffile')
            # middle lat lon
            lat = np.round(coords[128][128][1], 2)
            lon = np.round(coords[128][128][0], 2)
            geo_data_dict['Latitude'].append(lat)
            geo_data_dict['Longitude'].append(lon)
        except Exception as e:
            print(truth_fn)
            print(e)

sort_fns(data_dict, 'test')
print('done with test')
sort_fns(data_dict, 'val')
print('done with val')
sort_fns(data_dict, 'train')

with open('dict_lat_lon_fn.pkl', 'wb') as handle:
    pickle.dump(geo_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

