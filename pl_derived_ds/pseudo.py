import shutil
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

global data_par_dir
data_par_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/PL_SmokeViz/'
global full_data_dir
full_data_dir = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/full_dataset/'

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

def run_model(idx, yr, dn):
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
        bad_fn = "{}low_iou/{}".format(data_par_dir, fn)
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

    chkpt_path = './models/ckpt.pth'
    checkpoint = torch.load(chkpt_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_fn = get_best_file(test_loader, model)
    return best_fn

def get_indices(yr, dn):
    file_list = glob.glob('{}truth/{}/*/{}/*.tif'.format(full_data_dir, yr, dn))
    indices = []
    for fn in file_list:
        idx = fn.split('_')[-1].split('.')[0]
        indices.append(idx)
    indices = list(set(indices))
    indices.sort()
    return indices

def mv_files(truth_src, yr, dn, idx):
    truth_dst = truth_src.replace(full_data_dir, data_par_dir)
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
    indices = get_indices(yr, dn)
    print(indices)
    for idx in indices:
        best_fn = run_model(idx, yr, dn)
        if best_fn:
            mv_files(best_fn, yr, dn, idx)

