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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_iou(high_iou, med_iou, low_iou):
    intersection = high_iou[0] + med_iou[0] + low_iou[0]
    union = high_iou[1] + med_iou[1] + low_iou[1]
    print("high IoU: {}".format(high_iou[0]/high_iou[1]))
    print("med IoU: {}".format(med_iou[0]/med_iou[1]))
    print("low IoU: {}".format(low_iou[0]/low_iou[1]))
    overall_iou = intersection/union
    print("overall IoU: {}".format(overall_iou))
    return overall_iou, high_iou[0]/high_iou[1]

class IoUCalculator(object):
    """Computes and stores the current IoU and intersection and union sums"""
    def __init__(self, density):
        self.density = density
        self.IoU = 0
        self.reset()

    def reset(self):
        self.intersection = 0
        self.union = 0

    def update(self, pred, truth):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        curr_int = (pred + truth == 2).sum()
        curr_union = (pred + truth >= 1).sum()
        self.intersection += curr_int
        self.union += curr_union

    def all_reduce(self):
        rank = torch.device(f"cuda:{dist.get_rank()}")
        total = torch.tensor([self.intersection, self.union], dtype=torch.float32, device=rank)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        return total

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def val_model(dataloader, model, criterion, rank):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    high_iou = IoUCalculator('high')
    med_iou = IoUCalculator('medium')
    low_iou = IoUCalculator('low')
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(rank, dtype=torch.float32, non_blocking=True), batch_labels.to(rank, dtype=torch.float32, non_blocking=True)
        preds = model(batch_data)
        high_loss = criterion(preds[:,0,:,:], batch_labels[:,0,:,:]).to(rank)
        med_loss = criterion(preds[:,1,:,:], batch_labels[:,1,:,:]).to(rank)
        low_loss = criterion(preds[:,2,:,:], batch_labels[:,2,:,:]).to(rank)
        loss = 3*high_loss + 2*med_loss + low_loss
        test_loss = loss.item()
        total_loss += test_loss
        high_iou.update(preds[:,0,:,:], batch_labels[:,0,:,:])
        med_iou.update(preds[:,1,:,:], batch_labels[:,1,:,:])
        low_iou.update(preds[:,2,:,:], batch_labels[:,2,:,:])
    final_loss = total_loss/len(dataloader)
    loss_tensor = torch.tensor([final_loss]).to(rank)
    dist.all_reduce(loss_tensor)
    loss_tensor /= 8
    if rank==0:
        print("Validation Loss: {}".format(loss_tensor[0]), flush=True)
    return final_loss, high_iou.all_reduce(), med_iou.all_reduce(), low_iou.all_reduce()

def load_model(ckpt_loc, use_ckpt, use_recent, rank, cfg, exp_num):

    arch = cfg['architecture']
    encoder = cfg['encoder']
    lr = cfg['lr']

    #if encoder_weights == 'None':
    #    encoder_weights = None

    model = smp.create_model( # create any model architecture just with parameters, without using its class
            arch=arch,
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3, # model input channels
            classes=3 # model output channels
    )
    model = model.to(rank)

    if rank == 0:
        print(summary(model, input_size=(8,3,256,256)))

    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    start_epoch = 0
    best_loss = 0
    ckpt_pth = None

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if use_ckpt:
        if use_recent:
            ckpt_list = glob.glob('{}{}_{}_exp{}_*.pth'.format(ckpt_loc, arch, encoder, exp_num))
            ckpt_list.sort() # sort by time
            if ckpt_list:
                most_recent = ckpt_list.pop()
                ckpt_pth = most_recent
        else:
            ckpt_pth = ckpt_loc
        if ckpt_pth:
            if rank == 0:
                print('using this checkpoint: ', ckpt_pth)
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint=torch.load(ckpt_pth, map_location=map_location, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']

    return model, optimizer, start_epoch, best_loss


def train_model(train_dataloader, model, criterion, optimizer, rank):
    total_loss = 0.0
    model.train()
    torch.set_grad_enabled(True)
    start = time.time()
    for data in train_dataloader:

        optimizer.zero_grad() # zero the parameter gradients
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(rank, dtype=torch.float32, non_blocking=True), batch_labels.to(rank, dtype=torch.float32, non_blocking=True)

        preds = model(batch_data)

        high_loss = criterion(preds[:,0,:,:], batch_labels[:,0,:,:]).to(rank)
        med_loss = criterion(preds[:,1,:,:], batch_labels[:,1,:,:]).to(rank)
        low_loss = criterion(preds[:,2,:,:], batch_labels[:,2,:,:]).to(rank)
        loss = 6*high_loss + 4*med_loss + low_loss
        total_loss += loss.item()

        # compute gradient and do step
        loss.backward()
        optimizer.step()

    epoch_loss = total_loss/len(train_dataloader)
    if rank==0:
        print('training time: ', np.round(time.time() - start, 2), flush=True)
        print("training loss:   {}".format(np.round(epoch_loss,4)), flush=True)

def get_subset_train(data_dict):
    subset_data_dict = {'train':{'data':[], 'truth':[]}}
    num_samples = int(len(data_dict['train']['truth'])/5)
    subset_data_dict['train']['data'] = random.sample(data_dict['train']['data'], num_samples)
    subset_data_dict['train']['truth'] = random.sample(data_dict['train']['truth'], num_samples)
    return subset_data_dict

def get_transforms(train_augs):
    transform_list = [transforms.ToTensor()]
    if 'rhf' in train_augs.keys():
        transform_list.append(transforms.RandomHorizontalFlip(p=train_augs['rhf']))
    if 'rvf' in train_augs.keys():
        transform_list.append(transforms.RandomVerticalFlip(p=train_augs['rvf']))
    data_transforms = transforms.Compose(transform_list)
    return data_transforms

def prepare_dataloader(rank, world_size, data_dict, cat, batch_size, pin_memory=True, num_workers=4, is_train=True, train_aug=None):
    #if is_train:
    #    data_transforms = get_transforms(train_aug)
    #    dataset = SmokeDataset(data_dict[cat], transform=data_transforms)
    #else:
    data_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = SmokeDataset(data_dict[cat], transform=data_transforms)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=sampler)
    return dataloader


def set_seed(rank):
    seed = 0 
    torch.manual_seed(seed + rank)  # set each rank a different seed 
    torch.cuda.manual_seed_all(seed + rank)
    cudnn.deterministic = True  
    cudnn.benchmark = False  
    return


def main(rank, world_size, config_fn):
    set_seed(rank)
    torch.manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = False

    #exp_num = config_fn.split('exp')[-1].split('.json')[0]
    exp_num = '0'
    with open(config_fn) as fn:
        cfg = json.load(fn)
    arch = cfg['architecture']
    encoder = cfg['encoder']
    lr = cfg['lr']


    data_fn = cfg['datapointer']
    with open(data_fn, 'rb') as handle:
        data_dict = pickle.load(handle)

    n_epochs = 100
    start_epoch = 0
    lr = cfg['lr']
    batch_size = int(cfg['batch_size'])
    num_workers = int(cfg['num_workers'])
    encoder_weights = cfg['encoder_weights']

    setup(rank, world_size)


    if rank==0:
        print('data dict:              ', data_fn)
        print('config fn:              ', config_fn)
        print('number of train samples:', len(data_dict['train']['truth']))
        print('number of val samples:  ', len(data_dict['val']['truth']))
        print('number of test samples:  ', len(data_dict['test']['truth']))
        print('learning rate:          ', lr)
        print('batch_size:             ', batch_size)
        print('arch:                   ', arch)
        print('encoder:                ', encoder)
        print('num workers:            ', num_workers)
        print('num gpus:               ', world_size)

    use_ckpt = False
    use_recent = False
    use_ckpt = True
    #use_recent = True
    ckpt_save_loc = './Mie_models/'
    ckpt_loc = None
    if use_ckpt:
        if use_recent:
            ckpt_loc = ckpt_save_loc
        else:
            ckpt_loc = cfg['ckpt']

    model, optimizer, start_epoch, best_loss = load_model(ckpt_loc, use_ckpt, use_recent, rank, cfg, exp_num)

    criterion = nn.BCEWithLogitsLoss().to(rank)

    prev_iou = 0

    train_loader = prepare_dataloader(rank, world_size, data_dict, 'train', batch_size=batch_size, num_workers=num_workers, train_aug=cfg['train_augmentations'])
    val_loader = prepare_dataloader(rank, world_size, data_dict, 'val', batch_size=batch_size, is_train=False, num_workers=num_workers)

    for epoch in range(start_epoch, n_epochs):

        if rank==0:
            print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)
            start = time.time()

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        train_model(train_loader, model, criterion, optimizer, rank)
        val_loss, high_iou, med_iou, low_iou = val_model(val_loader, model, criterion, rank)

        if rank==0:
            print("time to run epoch:", np.round(time.time() - start, 2))
            iou, high_iou = get_iou(high_iou, med_iou, low_iou)
            if iou > prev_iou and iou > .40 and high_iou > .3:
                checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        'iou': iou
                        }
                ckpt_pth = '{}{}_{}_exp{}_{}.pth'.format(ckpt_save_loc, arch, encoder, exp_num, int(time.time()))
                torch.save(checkpoint, ckpt_pth)
                print('SAVING MODEL:\n', ckpt_pth, flush=True)
                prev_iou = iou

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 2 # num gpus
    if len(sys.argv) < 2:
        print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
    config_fn = str(sys.argv[1])
    mp.spawn(main, args=(world_size, config_fn), nprocs=world_size, join=True)

