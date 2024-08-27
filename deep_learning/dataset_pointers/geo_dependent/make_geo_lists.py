import pickle
import skimage

with open('filtered.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

geo_data_dict = {'NW':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                'SW':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                'NE':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                'SE':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}}
                }

                    
def get_section(lat, lon):
    if lat > 40 and lon < -100:
        quad = 'NW'
    if lat < 40 and lon < -100:
        quad = 'SW'
    if lat > 40 and lon > -100:
        quad = 'NE'
    if lat < 40 and lon > -100:
        quad = 'SE'
    return quad

def sort_fns(ds, split):
    truth_fns = ds[split]['truth'] 
    data_fns = ds[split]['data'] 
    
    for idx, truth_fn in enumerate(truth_fns):
        coords_fn = truth_fn.replace('truth', 'coords')
        coords = skimage.io.imread(coords_fn, plugin='tifffile')
        # middle lat lon
        lat = coords[128][128][0]
        lon = coords[128][128][1]
        quad = get_section(lat, lon)
        geo_data_dict[quad][split]['truth'].append(truth_fn)
        geo_data_dict[quad][split]['data'].append(data_fns[idx])
    
sort_fns(data_dict, 'test')
print('original test dataset size: ', len(data_dict['test']['truth']))
print('for testing datasets:')
print('NW: ', len(geo_data_dict['NW']['test']['truth']))
print('SW: ', len(geo_data_dict['SW']['test']['truth']))
print('NE: ', len(geo_data_dict['NE']['test']['truth']))
print('SE: ', len(geo_data_dict['SE']['test']['truth']))
#sort_fns(data_dict, 'train')
#sort_fns(data_dict, 'val')

def save_data_dicts(quad):
    with open('{}.pkl'.format(quad), 'wb') as handle:
        pickle.dump(geo_data_dict[quad], handle, protocol=pickle.HIGHEST_PROTOCOL)

for quad in geo_data_dict.keys():
    save_data_dicts(quad)
