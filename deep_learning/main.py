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

def get_iou(high_iou, med_iou, low_iou):
    intersection = high_iou[0] + med_iou[0] + low_iou[0]
    union = high_iou[1] + med_iou[1] + low_iou[1]
    overall_iou = intersection/union
    print("Overall IoU all density smoke: {}".format(overall_iou))
    return overall_iou

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
        if curr_union > 0:
            curr_IoU = curr_int / curr_union
        self.intersection += curr_int
        self.union += curr_union

    def all_reduce(self):
        rank = torch.device(f"cuda:{dist.get_rank()}")
        total = torch.tensor([self.intersection, self.union], dtype=torch.float32, device=rank)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        print("IoU for {} density smoke: {}".format(self.density, self.intersection/self.union))
        return self.intersection, self.union

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
    if rank==0:
        print("Validation Loss: {}".format(np.round(final_loss,4)), flush=True)
    return final_loss, high_iou.all_reduce(), med_iou.all_reduce(), low_iou.all_reduce()



def load_model(ckpt_loc, use_ckpt, use_recent, rank, arch, encoder, lr, exp_num, encoder_weights):

    if encoder_weights == 'None':
        encoder_weights = None
    model = smp.create_model( # create any model architecture just with parameters, without using its class
            arch=arch,
            encoder_name=encoder,
            #encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
            encoder_weights=encoder_weights, # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3, # model input channels
            classes=3, # model output channels
    )

    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    start_epoch = 0
    best_loss = 0
    ckpt_pth = None

    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if use_ckpt:
        if use_recent:
            ckpt_list = glob.glob('{}{}_exp{}_*.pth'.format(ckpt_loc, arch, exp_num))
            ckpt_list.sort()
            if ckpt_list:
                # sorted by time
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
        loss = 3*high_loss + 2*med_loss + low_loss
        total_loss += loss.item()

        # compute gradient and do step
        loss.backward()
        optimizer.step()

    epoch_loss = total_loss/len(train_dataloader)
    if rank==0:
        print('training time: ', np.round(time.time() - start, 2), flush=True)
        print("training loss:   {}".format(np.round(epoch_loss,4)), flush=True)

def prepare_dataloader(rank, world_size, data_dict, cat, batch_size, pin_memory=True, num_workers=4, is_train=True):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = SmokeDataset(data_dict[cat], data_transforms)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

def main(rank, world_size, config_fn):

    exp_num = config_fn.split('exp')[-1].split('.json')[0]
    with open(config_fn) as fn:
        hyperparams = json.load(fn)

    #with open('./dataset_pointers/make_list/subsample.pkl', 'rb') as handle:
    #with open('./dataset_pointers/both_sats/both_sats.pkl', 'rb') as handle:
    #with open('./dataset_pointers/smokeviz_yr_split/SmokeViz.pkl', 'rb') as handle:
    #data_fn = './dataset_pointers/smokeviz_yr_split/SmokeViz.pkl'
    #data_fn = './dataset_pointers/large/large.pkl'
    data_fn = './dataset_pointers/Mie/Mie.pkl'
    data_fn = hyperparams['datapointer']

    with open(data_fn, 'rb') as handle:
        data_dict = pickle.load(handle)

    n_epochs = 100
    start_epoch = 0
    arch = hyperparams['architecture']
    encoder = hyperparams['encoder']
    lr = hyperparams['lr']
    batch_size = int(hyperparams['batch_size'])
    num_workers = int(hyperparams['num_workers'])
    encoder_weights = hyperparams['encoder_weights']

    setup(rank, world_size)

    train_loader = prepare_dataloader(rank, world_size, data_dict, 'train', batch_size=batch_size, num_workers=num_workers)
    val_loader = prepare_dataloader(rank, world_size, data_dict, 'val', batch_size=batch_size, is_train=False, num_workers=num_workers)

    if rank==0:
        print('data dict:              ', data_fn)
        print('config fn:              ', config_fn)
        print('number of train samples:', len(data_dict['train']['truth']))
        print('number of val samples:  ', len(data_dict['val']['truth']))
        print('learning rate:          ', lr)
        print('batch_size:             ', batch_size)
        print('arch:                   ', arch)
        print('encoder:                ', encoder)
        print('num workers:            ', num_workers)
        print('num gpus:               ', world_size)
        print('pretrained weights:     ', encoder_weights)

    use_ckpt = False
    use_recent = False
    #use_ckpt = True
    #use_recent = True
    ckpt_save_loc = './models/'
    ckpt_loc = None
    if use_ckpt:
        if use_recent:
            ckpt_loc = ckpt_save_loc
        else:
            ckpt_loc = hyperparams['ckpt']

    model, optimizer, start_epoch, best_loss = load_model(ckpt_loc, use_ckpt, use_recent, rank, arch, encoder, lr, exp_num, encoder_weights)

    criterion = nn.BCEWithLogitsLoss().to(rank)

    prev_iou = 0

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
            iou = get_iou(high_iou, med_iou, low_iou)
            if iou > prev_iou:
                checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        'iou': iou
                        }
                ckpt_pth = '{}{}_exp{}_{}.pth'.format(ckpt_save_loc, arch, exp_num, int(time.time()))
                torch.save(checkpoint, ckpt_pth)
                print('SAVING MODEL:\n', ckpt_pth, flush=True)
                prev_iou = iou

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 8 # num gpus
    if len(sys.argv) < 2:
        print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
    config_fn = str(sys.argv[1])
    mp.spawn(main, args=(world_size, config_fn), nprocs=world_size, join=True)

