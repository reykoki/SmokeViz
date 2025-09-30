import pickle
import random
import torch.backends.cudnn as cudnn
import os
import glob
import time
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from SmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
import torch.multiprocessing as mp
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary

def get_metrics(high_metrics, med_metrics, low_metrics):
    intersection = high_metrics[0] + med_metrics[0] + low_metrics[0]
    union = high_metrics[1] + med_metrics[1] + low_metrics[1]
    overall_iou = intersection/union
    print("---")
    print("high IoU: {}".format(high_metrics[0]/high_metrics[1]))
    print("med IoU: {}".format(med_metrics[0]/med_metrics[1]))
    print("low IoU: {}".format(low_metrics[0]/low_metrics[1]))
    print("Overall IoU: {}".format(overall_iou))
    print("---")

    TP = high_metrics[2] + med_metrics[2] + low_metrics[2]
    TN = high_metrics[3] + med_metrics[3] + low_metrics[3]
    FP = high_metrics[4] + med_metrics[4] + low_metrics[4]
    FN = high_metrics[5] + med_metrics[5] + low_metrics[5]
    union = high_metrics[1] + med_metrics[1] + low_metrics[1]
    print("high recall: {}".format(high_metrics[2]/(high_metrics[2]+high_metrics[5])))
    print("med recall: {}".format(med_metrics[2]/(med_metrics[2]+med_metrics[5])))
    print("low recall: {}".format(low_metrics[2]/(low_metrics[2]+low_metrics[5])))
    print("Overall recall: {}".format(TP/(TP+FN)))
    print("---")

    print("high precision: {}".format(high_metrics[2]/(high_metrics[2]+high_metrics[4])))
    print("med precision: {}".format(med_metrics[2]/(med_metrics[2]+med_metrics[4])))
    print("low precision: {}".format(low_metrics[2]/(low_metrics[2]+low_metrics[4])))
    print("Overall precision: {}".format(TP/(TP+FP)))
    print("---")

    return overall_iou


class metrics_Calculator(object):
    """Computes and stores the current IoU and intersection and union sums"""
    def __init__(self, density):
        self.density = density
        self.reset()

    def reset(self):
        self.intersection = 0
        self.union = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def update(self, pred, truth):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        self.intersection += (pred + truth == 2).sum()
        self.union += (pred + truth >= 1).sum()
        self.TP += (pred + truth == 2).sum()
        self.TN += (pred + truth == 0).sum()
        self.FP += (truth - pred == -1).sum()
        self.FN += (pred - truth == -1).sum()

    def all_reduce(self):
        rank = torch.device(f"cuda:{dist.get_rank()}")
        metrics = torch.tensor([self.intersection, self.union, self.TP, self.TN, self.FP, self.FN], dtype=torch.float32, device=rank)
        dist.all_reduce(metrics, dist.ReduceOp.SUM, async_op=False)
        return metrics

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def val_model(dataloader, model, rank):
    model.eval()
    torch.set_grad_enabled(False)
    high_metrics = metrics_Calculator('high')
    med_metrics = metrics_Calculator('medium')
    low_metrics = metrics_Calculator('low')
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(rank, dtype=torch.float32, non_blocking=True), batch_labels.to(rank, dtype=torch.float32, non_blocking=True)
        preds = model(batch_data)
        high_metrics.update(preds[:,0,:,:], batch_labels[:,0,:,:])
        med_metrics.update(preds[:,1,:,:], batch_labels[:,1,:,:])
        low_metrics.update(preds[:,2,:,:], batch_labels[:,2,:,:])
    return high_metrics.all_reduce(), med_metrics.all_reduce(), low_metrics.all_reduce()


def load_model(ckpt_loc, use_ckpt, use_recent, rank, cfg, exp_num):

    arch = cfg['architecture']
    encoder = cfg['encoder']
    lr = cfg['lr']

    model = smp.create_model( # create any model architecture just with parameters, without using its class
            arch=arch,
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3, # model input channels
            classes=3 # model output channels
    )
    model = model.to(rank)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    start_epoch = 0
    best_loss = 0

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

#    if rank == 0:
#        print('using this checkpoint: ', ckpt_loc)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint=torch.load(ckpt_loc, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    return model, optimizer, start_epoch, best_loss


def get_transforms(train_augs):
    transform_list = [transforms.ToTensor()]
    if 'rhf' in train_augs.keys():
        transform_list.append(transforms.RandomHorizontalFlip(p=train_augs['rhf']))
    if 'rvf' in train_augs.keys():
        transform_list.append(transforms.RandomVerticalFlip(p=train_augs['rvf']))
    data_transforms = transforms.Compose(transform_list)
    return data_transforms

def prepare_dataloader(rank, world_size, data_dict, cat, batch_size, pin_memory=True, num_workers=4, is_train=True, train_aug=None):
    if is_train:
        data_transforms = get_transforms(train_aug)
        dataset = SmokeDataset(data_dict[cat], transform=data_transforms)
    else:
        data_transforms = transforms.Compose([transforms.ToTensor()])
        dataset = SmokeDataset(data_dict[cat], transform=data_transforms)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=sampler)
    return dataloader

def main(rank, world_size):

    setup(rank, world_size)
    threshes = ['005', '01', '05', '1', '15', '2', '25', '3', '35', '4']
    try:

        for model_thresh in threshes:
            for data_thresh in threshes:
                config_fn = "./configs_thresh/testing_1/pspnet_efficientnet-b2_{}.json".format(model_thresh)

                with open(config_fn) as fn:
                    cfg = json.load(fn)
                arch = cfg['architecture']
                encoder = cfg['encoder']

                data_fn = '/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/deep_learning/dataset_pointers/thresh/thresh_{}.pkl'.format(data_thresh)
                with open(data_fn, 'rb') as handle:
                    data_dict = pickle.load(handle)

                batch_size = int(cfg['batch_size'])
                num_workers = int(cfg['num_workers'])

                test_loader = prepare_dataloader(rank, world_size, data_dict, 'test', batch_size=batch_size, is_train=False, num_workers=num_workers)

                if rank==0:
                    #print('data dict:              ', data_fn)
                    #print('config fn:              ', config_fn)
                    print("training threshold: .{}".format(model_thresh))
                    print("testing threshold: .{}".format(data_thresh))
                    #print('number of test samples: ', len(data_dict['test']['truth']))

                use_ckpt = True
                use_recent = False
                ckpt_loc = cfg['ckpt']
                model, optimizer, start_epoch, best_loss = load_model(ckpt_loc, use_ckpt, use_recent, rank, cfg, model_thresh)

                if rank==0:
                    start = time.time()

                test_loader.sampler.set_epoch(start_epoch)

                high_metrics, med_metrics, low_metrics = val_model(test_loader, model, rank)

                if rank==0:
                    iou = get_metrics(high_metrics, med_metrics, low_metrics)
                    print("time to run testing:", np.round(time.time() - start, 2))
                    print("")

    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = False
    world_size = 2 # num gpus
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
