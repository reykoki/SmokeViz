import pickle
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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train_model(train_dataloader):
    for data in train_dataloader:
        pass


def prepare_dataloader(rank, world_size, data_dict, cat, batch_size, pin_memory=True, num_workers=0):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = SmokeDataset(data_dict[cat], data_transforms)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

def main(rank, world_size, config_fn):

    exp_num = config_fn.split('exp')[-1].split('.json')[0]
    with open('configs/{}'.format(config_fn)) as fn:
        hyperparams = json.load(fn)

    data_fn ='../dataset_pointers/large/large.pkl'
    #data_fn = '../dataset_pointers/smokeviz_yr_split/SmokeViz.pkl'

    with open(data_fn, 'rb') as handle:
        data_dict = pickle.load(handle)

    n_epochs = 1
    start_epoch = 0
    arch = hyperparams['architecture']
    lr = hyperparams['lr']

    setup(rank, world_size)
    if rank==0:
        print(data_fn)
        print('batch size: ', int(hyperparams['batch_size']))
    for num_workers in range(2,5,2):

        if rank==0:
            print("NUM WORKERS:", num_workers)

        train_loader = prepare_dataloader(rank, world_size, data_dict, 'train', batch_size=int(hyperparams['batch_size']), num_workers=num_workers)

        model = smp.create_model( # create any model architecture just with parameters, without using its class
                arch=arch,
                encoder_name=hyperparams['encoder'],
                #encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3, # model input channels
                classes=3, # model output channels
        )
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        for epoch in range(start_epoch, n_epochs):
            if rank==0:
                start = time.time()
            train_loader.sampler.set_epoch(epoch)
            train_model(train_loader)
            if rank==0:
                print("time to run epoch:", np.round(time.time() - start, 4))

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 8 # num gpus
    if len(sys.argv) < 2:
        print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
    config_fn = str(sys.argv[1])
    mp.spawn(main, args=(world_size, config_fn), nprocs=world_size, join=True)

