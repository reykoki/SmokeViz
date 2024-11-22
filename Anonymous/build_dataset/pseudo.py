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
global pseudo_dir
pseudo_dir = "$PSEUDO_DS_DIR"
global full_data_dir
full_data_dir = "$FULL_DS_DIR"

def sample_doesnt_exist(yr, dn, idx, density):
    fn_head_parts = fn_head.split('_')
    sat_num = fn_head_parts[0]
    start_scan = fn_head_parts[1]
    file_list = glob.glob('{}truth/{}/{}/{}/*_{}_*_{}.tif'.format(PL_dir, yr, density, dn, sat_num, start_scan, idx))
    if len(file_list) > 0:
        print("FILES THAT ALREADY EXIST:", file_list, flush=True)
        if len(file_list) > 1:
            file_list.sort()
            fns_to_rm = file_list[1:]
            for fn in fns_to_rm:
                os.remove(fn)
                os.remove(fn.replace('truth','data'))
        return False
    bad_file_list = glob.glob('{}low_iou/{}_{}_*_{}.tif'.format(PL_dir, sat_num, start_scan, idx))

    if len(bad_file_list) > 0:
        print("BAD FILES:", bad_file_list, flush=True)
        return False
    return True

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

@ray.remote(max_calls=1)
def run_model(idx_info):
    idx = idx_info['idx']
    yr = idx_info['yr']
    dn = idx_info['dn']


    data_dict = get_file_list(idx, yr, dn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transforms = transforms.Compose([transforms.ToTensor()])

    test_set = SmokeDataset(data_dict['find'], data_transforms)

    print('there are {} images for this annotation'.format(len(test_set)))

    def get_best_file(dataloader, model):
        model.eval()
        torch.set_grad_enabled(False)
        # iou has to be more than .01
        best_iou = .01
        ious = []
        best_truth_fn = None
        for idx, data in enumerate(dataloader):
            batch_data, batch_labels, truth_fn = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            preds = model(batch_data)
            iou = compute_iou(preds, batch_labels)
            ious.append(iou)
            if iou > best_iou:
                best_iou = iou
                best_truth_fn = truth_fn
        print("IoU scores: {}".format(ious, flush=True))
        best_iou_idx = ious.index(max(ious))
        print('best IoU index', best_iou_idx)
        if best_truth_fn:
            print('best_truth_fn: ', best_truth_fn)
            return  best_truth_fn[0]

        fn = truth_fn[0].split('/')[-1]
        idx = fn.split('_')[-1].split('.')[0]
        bad_fn = "{}low_iou/{}".format(PL_dir, '{}{}_{}'.format(yr, dn, idx))

        with open(bad_fn, 'w') as fp:
            pass
        return None


    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    model = smp.DeepLabV3Plus(
            encoder_name="timm-efficientnet-b2",
            encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3, # model input channels
            classes=3, # model output channels
    )
    model = model.to(device)

    chkpt_path = './models/ckpt2.pth'
    checkpoint = torch.load(chkpt_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_fn = get_best_file(test_loader, model)
    if best_fn:
        mv_files(best_fn, yr, dn, idx)
    return

def indices_dont_exist(yr, dn, indices):
    day_list = []
    for idx in indices:
        file_list = glob.glob('{}truth/{}/*/{}/*_{}.tif'.format(PL_dir, yr, dn, idx))
        if len(file_list) > 0:
            print("PSEUDO THAT ALREADY EXIST:", file_list, flush=True)
        low_iou_file_list = glob.glob('{}low_iou/{}{}_{}'.format(PL_dir, yr, dn, idx))
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

def mv_files(truth_src, yr, dn, idx):
    truth_dst = truth_src.replace(full_data_dir, PL_dir)
    data_src = truth_src.replace('truth','data')
    data_dst = truth_dst.replace('truth','data')

    if not os.path.exists(truth_dst):
        os.symlink(truth_src, truth_dst)
    if not os.path.exists(data_dst):
        os.symlink(data_src, data_dst)
    #shutil.copyfile(truth_src, truth_dst)
    #shutil.copyfile(data_src, data_dst)
    #if os.path.exists(truth_dst) and os.path.exists(data_dst):
    #    for f in glob.glob(dn_dir+'*/*/*/*_{}.tif'.format(idx)):
    #        os.remove(f)

def find_best_data(yr, dn):
    start = time.time()
    idx_list = get_indices(yr, dn)
    if idx_list:
        print(idx_list)
        ray_dir = "{}{}{}pseudo".format(ray_par_dir,yr,dn)
        if not os.path.isdir(ray_dir):
            os.mkdir(ray_dir)
        ray.init(num_cpus=10, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1', object_store_memory=10**9)
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
    main(start_dn, end_dn, yr)
