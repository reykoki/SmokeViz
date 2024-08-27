import pickle
from calendar import monthrange
import skimage

with open('../geo_dependent/filtered.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

month_data_dict = {'1':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '2':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '3':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '4':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '5':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '6':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '7':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '8':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '9':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '10':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '11':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}},
                '12':
                    {'train': {'truth': [], 'data': []},
                     'val': {'truth': [], 'data': []},
                     'test': {'truth': [], 'data': []}}
                }


def sort_fns(ds, split):
    truth_fns = ds[split]['truth']
    data_fns = ds[split]['data']
    curr_dn = 0
    for month in month_data_dict.keys():
        days = monthrange(int(2022), int(month))
        for day in range(1,days[1]+1):
            curr_dn += 1
            dn_str = '_s2022' + str(curr_dn).zfill(3)
            truth_fns_dn = [i for i in truth_fns if dn_str in i]
            month_data_dict[month][split]['truth'].extend(truth_fns_dn)
        data_fns_month = [fn.replace('truth', 'data') for fn in month_data_dict[month][split]['truth']]
        month_data_dict[month][split]['data'].extend(data_fns_month)
    return month_data_dict


sort_fns(data_dict, 'test')
print('1: ', len(month_data_dict['1']['test']['truth']))
print('2: ', len(month_data_dict['2']['test']['truth']))
print('3: ', len(month_data_dict['3']['test']['truth']))
print('4: ', len(month_data_dict['4']['test']['truth']))
print('5: ', len(month_data_dict['5']['test']['truth']))
print('6: ', len(month_data_dict['6']['test']['truth']))
print('7: ', len(month_data_dict['7']['test']['truth']))
print('8: ', len(month_data_dict['8']['test']['truth']))
#sort_fns(data_dict, 'train')
#sort_fns(data_dict, 'val')

def save_data_dicts(month):
    with open('{}.pkl'.format(month), 'wb') as handle:
        pickle.dump(month_data_dict[month], handle, protocol=pickle.HIGHEST_PROTOCOL)

for month in month_data_dict.keys():
    save_data_dicts(month)
