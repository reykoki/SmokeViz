import pickle

with open('../make_list/pseudo_labeled.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

sat_data_dict = {'G17':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                'G16':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                }

def sort_fns(ds, split):
    truth_fns = ds[split]['truth']
    for sat in sat_data_dict.keys():
        print(sat)
        truth_fns_sat = [i for i in truth_fns if sat in i]
        sat_data_dict[sat][split]['truth'].extend(truth_fns_sat)
        data_fns_sat = [fn.replace('truth', 'data') for fn in truth_fns_sat]
        sat_data_dict[sat][split]['data'].extend(data_fns_sat)

sort_fns(data_dict, 'test')
print('original test dataset size: ', len(data_dict['test']['truth']))
print('for testing datasets:')
print('G16: ', len(sat_data_dict['G16']['test']['truth']))
print('G17: ', len(sat_data_dict['G17']['test']['truth']))
#sort_fns(data_dict, 'train')
#sort_fns(data_dict, 'val')

def save_data_dicts(sat):
    with open('{}.pkl'.format(sat), 'wb') as handle:
        pickle.dump(sat_data_dict[sat], handle, protocol=pickle.HIGHEST_PROTOCOL)

for sat in sat_data_dict.keys():
    save_data_dicts(sat)
