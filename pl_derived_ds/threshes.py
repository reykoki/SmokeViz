import shutil
import pickle
import sys
import os
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from SmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
import ray
import numpy as np
import time

global ray_par_dir
ray_par_dir = "/tmp/"
global full_data_dir
full_data_dir = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/full_dataset/'

def get_file_list(idx, yr, dn):
    truth_file_list = []
    truth_file_list = glob.glob('{}truth/{}/*/{}/*_{}.tif'.format(full_data_dir, yr, dn, idx))
    truth_file_list.sort()
    data_file_list = [s.replace('truth','data') for s in truth_file_list]
    #print('number of samples for idx:', len(truth_file_list))
    data_dict = {'find': {'truth': truth_file_list, 'data': data_file_list}}
    return data_dict

def compute_iou(preds, truths):
    densities = ['heavy', 'medium', 'low']
    intersection = 0
    union = 0
    for idx, level in enumerate(densities):
        pred = preds[:,idx,:,:]
        true = truths[:,idx,:,:]
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        intersection += (pred + true == 2).sum()
        union += (pred + true >= 1).sum()
    try:
        iou = intersection / union
        return iou
    except Exception as e:
        print(e)
    return 0

def create_idx_PLDR_dict(dataloader, model, smoke_idx, device):
    torch.set_grad_enabled(False)
    idx_PLDR_dict = {smoke_idx: {'fns': [], 'IoUs': [], 'best_IoU_fn': None, 'best_IoU': 0}}
    for dl_idx, data in enumerate(dataloader):
        batch_data, batch_labels, truth_fn = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)
        iou = compute_iou(preds, batch_labels)
        idx_PLDR_dict[smoke_idx]['IoUs'].append(iou.item())
        idx_PLDR_dict[smoke_idx]['fns'].append(truth_fn[0])
        if iou > idx_PLDR_dict[smoke_idx]['best_IoU']:
            idx_PLDR_dict[smoke_idx]['best_IoU'] = iou.item()
            idx_PLDR_dict[smoke_idx]['best_IoU_fn'] = truth_fn[0]
    #print("IoU scores: {}".format(idx_PLDR_dict[smoke_idx]['IoUs']), flush=True)
    return idx_PLDR_dict

@ray.remote(max_calls=1)
def run_model(idx_info):
    smoke_idx = idx_info['idx']
    yr = idx_info['yr']
    dn = idx_info['dn']
    data_dict = get_file_list(smoke_idx, yr, dn)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    data_transforms = transforms.Compose([transforms.ToTensor()])
    test_set = SmokeDataset(data_dict['find'], data_transforms)
    print('there are {} images for annotation idx {}'.format(len(test_set), smoke_idx))
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    model = smp.create_model(
            arch="PSPNet",
            #arch="DeepLabV3Plus",
            encoder_name="timm-efficientnet-b2",
            encoder_weights=None,
            in_channels=3, # model input channels
            classes=3 # model output channels
    )

    model = model.to(device)
    model.eval()
    chkpt_path = './models/PSPNet_Mie.pth'
    #chkpt_path = './models/ckpt2.pth'
    checkpoint = torch.load(chkpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    idx_PLDR_dict = create_idx_PLDR_dict(test_loader, model, smoke_idx, device)
    return idx_PLDR_dict

def indices_dont_exist(yr, dn, indices):
    day_list = []
    pkl_fn = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/PLDR_IoUs/round1/PLDR_{}_{}.pkl".format(yr, dn)
    if os.path.exists(pkl_fn) is False:
        for idx in indices:
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

def find_best_data(yr, dn):
    start = time.time()
    idx_list = get_indices(yr, dn)
    if idx_list:
        #print(idx_list)
        ray_dir = "{}{}{}pseudo".format(ray_par_dir,yr,dn)
        if not os.path.isdir(ray_dir):
            os.mkdir(ray_dir)
        ray.init(num_cpus=24, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True)
        list_of_idx_PLDR_dicts = ray.get([run_model.remote(idx_dict) for idx_dict in idx_list])
        ray.shutdown()
        if os.path.exists(ray_dir):
            shutil.rmtree(ray_dir)
        PLDR_dict = {}
        for idx_PLDR_dict in list_of_idx_PLDR_dicts:
            smoke_idx = list(idx_PLDR_dict.keys())[0]
            print(smoke_idx)
            print(idx_PLDR_dict[smoke_idx]['IoUs'])
            print(idx_PLDR_dict[smoke_idx]['best_IoU_fn'])
            print(idx_PLDR_dict[smoke_idx]['best_IoU'])
            PLDR_dict.update(idx_PLDR_dict)
        pkl_fn = "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz_datasets/PLDR_IoUs/round1/PLDR_{}_{}.pkl".format(yr, dn)
        with open(pkl_fn, 'wb') as f:
            pickle.dump(PLDR_dict, f)
    print("Time elapsed for pseudo labeling for day {}{}: {}s".format(yr, dn, int(time.time() - start)), flush=True)

def main(start_dn, end_dn, yr):
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    print(dns)
    #dns.reverse()
    for dn in dns:
        dn = str(dn).zfill(3)
        find_best_data(yr, dn)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
