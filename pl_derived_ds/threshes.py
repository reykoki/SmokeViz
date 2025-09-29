import shutil
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
full_data_dir = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/full_dataset/'

def get_file_list(idx, yr, dn):
    truth_file_list = []
    truth_file_list = glob.glob('{}truth/{}/*/{}/*_{}.tif'.format(full_data_dir, yr, dn, idx))
    truth_file_list.sort()
    print(truth_file_list)
    data_file_list = [s.replace('truth','data') for s in truth_file_list]
    print('number of samples for idx:', len(truth_file_list))
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

def create_idx_PLDR_dict(dataloader, model, smoke_idx):
    model.eval()
    torch.set_grad_enabled(False)
    idx_PLDR_dict = {smoke_idx: {'fns':[], 'IoUs' [], 'best_IoU_fn': None, 'best_IoU': 0}}
    for dl_idx, data in enumerate(dataloader):
        batch_data, batch_labels, data_fn = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)
        iou = compute_iou(preds, batch_labels)
        idx_PLDR_dict[smoke_idx]['IoUs'].append(iou)
        idx_PLDR_dict[smoke_idx]['fns'].append(data_fn)
        if iou > idx_PLDR_dict[smoke_idx]['best_IoU']:
            idx_PLDR_dict[smoke_idx]['best_IoU'] = iou
            idx_PLDR_dict[smoke_idx]['best_IoU_fn'] = data_fn[0]
    print("IoU scores: {}".format(idx_PLDR_dict[smoke_idx]['IoUs'], flush=True))
    return idx_PLDR_dict

@ray.remote(max_calls=1)
def run_model(idx_info):
    idx = idx_info['idx']
    yr = idx_info['yr']
    dn = idx_info['dn']

    data_dict = get_file_list(idx, yr, dn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([transforms.ToTensor()])
    test_set = SmokeDataset(data_dict['find'], data_transforms)
    print('there are {} images for this annotation'.format(len(test_set)))
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    model = smp.DeepLabV3Plus(
            encoder_name="timm-efficientnet-b2",
            encoder_weights=None,
            in_channels=3,
            classes=3,
    )
    model = model.to(device)
    chkpt_path = './models/ckpt2.pth'
    checkpoint = torch.load(chkpt_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    idx_PLDR_dict = create_idx_PLDR_dict(test_loader, model, smoke_idx)
    return idx_PLDR_dict

def indices_dont_exist(yr, dn, indices):
    day_list = []
    for idx in indices:
        file_list = glob.glob('{}truth/{}/*/{}/*_{}.tif'.format(master_PL_dir, yr, dn, idx))
        if len(file_list) > 0:
            print("PSEUDO THAT ALREADY EXIST:", file_list, flush=True)
        low_iou_file_list = glob.glob('{}low_iou/{}{}_{}'.format(master_PL_dir, yr, dn, idx))
        if len(low_iou_file_list) > 0:
            print("LOW IOU FILE:", low_iou_file_list, flush=True)
        if len(file_list) == 0 and len(low_iou_file_list) == 0:
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

def mv_files(truth_src, yr, dn, curr_PL_dir):
    truth_dst = truth_src.replace(full_data_dir, curr_PL_dir)
    data_src = truth_src.replace('truth','data')
    data_dst = truth_dst.replace('truth','data')
    if not os.path.exists(truth_dst):
        os.symlink(truth_src, truth_dst)
    if not os.path.exists(data_dst):
        os.symlink(data_src, data_dst)

def find_best_data(yr, dn):
    start = time.time()
    idx_list = get_indices(yr, dn)
    if idx_list:
        print(idx_list)
        ray_dir = "{}{}{}pseudo".format(ray_par_dir,yr,dn)
        if not os.path.isdir(ray_dir):
            os.mkdir(ray_dir)
        ray.init(num_cpus=24, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1', object_store_memory=10**9)
        ray.get([run_model.remote(idx_dict) for idx_dict in idx_list])
        ray.shutdown()
        shutil.rmtree(ray_dir)
    print("Time elapsed for pseudo labeling for day {}{}: {}s".format(yr, dn, int(time.time() - start)), flush=True)

def main(start_dn, end_dn, yr):
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    #dns.reverse()
    for dn in dns:
        dn = str(dn).zfill(3)
        find_best_data(yr, dn)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    global master_PL_dir
    master_PL_dir = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/PL_25/'

    main(start_dn, end_dn, yr)
