import os
import pickle
import glob
import random

yrs = ['2018', '2019', '2020', '2021', '2023']
val_test_yr = '2022'

truth_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/truth/'
#truth_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/new_data/truth/'

cats = ['Light', 'Medium', 'Heavy', 'None']
#cat_count = {'Light':100, 'Medium':100, 'Heavy': 100, 'None': 100}
#cat_count = {'Light':1e10, 'Medium': 1e10, 'Heavy': 1e10, 'None': 1e10}
cat_count = {'Light':int(1e3), 'Medium': int(1e3), 'Heavy': int(1e3), 'None': int(1e3)}
cat_count = {'Light':1e10, 'Medium': 1e10, 'Heavy': 1e10}


def list_every_ten_days(dataset):
    dns = list(range(0,37))
    dns_filled = [str(item).zfill(2) for item in dns]
    if dataset == 'val':
        val_days = dns_filled[::2]
        return val_days
    if dataset == 'test':
        test_days = dns_filled[1::2]
        return test_days

def get_val_test_fns(yr, cat_count, dataset):
    truth_fns = []
    data_fns = []
    days_oi = list_every_ten_days(dataset)
    yr_truth_dir = truth_dir + yr + '/'
    cat_num_files = 0
    for cat in cat_count:
        cat_truth_fns = []
        for days in days_oi:
            cat_truth_fns.extend(glob.glob('{}{}/*_s{}{}*.tif'.format(yr_truth_dir, cat, yr, days)))
        if len(cat_truth_fns) > cat_count[cat]:
            random.shuffle(cat_truth_fns)
            cat_truth_fns = cat_truth_fns[:cat_count[cat]]
        cat_data_fns = [s.replace('truth','data') for s in cat_truth_fns]
        truth_fns.extend(cat_truth_fns)
        data_fns.extend(cat_data_fns)
    return truth_fns, data_fns

def get_train_fns(yrs, cat_count):
    truth_fns = []
    data_fns = []
    for yr in yrs:
        yr_truth_dir = truth_dir + yr + '/'
        cat_num_files = 0
        for cat in cat_count:
            cat_truth_fns = glob.glob('{}{}/*.tif'.format(yr_truth_dir, cat))
            if len(cat_truth_fns) > cat_count[cat]:
                random.shuffle(cat_truth_fns)
                cat_truth_fns = cat_truth_fns[:cat_count[cat]]
            cat_data_fns = [s.replace('truth','data') for s in cat_truth_fns]
            truth_fns.extend(cat_truth_fns)
            data_fns.extend(cat_data_fns)
    return truth_fns, data_fns

train_truth, train_data_fns = get_train_fns(yrs, cat_count)
val_truth, val_data_fns = get_val_test_fns(val_test_yr, cat_count, 'val')
test_truth, test_data_fns = get_val_test_fns(val_test_yr, cat_count, 'test')

data_dict = {'train': {'truth': train_truth, 'data': train_data_fns},
             'val': {'truth': val_truth, 'data': val_data_fns},
             'test': {'truth': test_truth, 'data': test_data_fns}}

print('number of train samples:', len(train_truth))
print('number of val samples:', len(val_truth))
print('number of test samples:', len(test_truth))

with open('pseudo_labeled.pkl', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

