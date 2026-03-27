import pickle
import random
import torch.backends.cudnn as cudnn
import os
import glob
import time
import sys
import json
import csv
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


def log_results(log_path, row):
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_iou(high_iou, med_iou, low_iou):
    print("high IoU: {}".format(high_iou[0]/high_iou[1]))
    print("med IoU: {}".format(med_iou[0]/med_iou[1]))
    print("low IoU: {}".format(low_iou[0]/low_iou[1]))
    intersection = high_iou[0] + med_iou[0] + low_iou[0]
    union = high_iou[1] + med_iou[1] + low_iou[1]
    overall_iou = intersection/union
    print("overall IoU: {}".format(overall_iou))
    return overall_iou, high_iou[0]/high_iou[1]


class IoUCalculator(object):
    """Computes and stores the current IoU and intersection and union sums"""
    def __init__(self, density):
        self.density = density
        self.reset()

    def reset(self):
        device = torch.device(f"cuda:{dist.get_rank()}")
        self.intersection = torch.tensor(0.0, device=device)
        self.union = torch.tensor(0.0, device=device)

    def update(self, pred, truth):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        curr_int = (pred + truth == 2).sum()
        curr_union = (pred + truth >= 1).sum()
        self.intersection += curr_int
        self.union += curr_union

    def all_reduce(self):
        total = torch.stack([self.intersection, self.union])
        dist.all_reduce(total, dist.ReduceOp.SUM)
        return total


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def combined_loss(preds, labels, bce_criterion, dice_criterion, high_w=3, med_w=2):
    high_bce  = bce_criterion(preds[:,0,:,:], labels[:,0,:,:])
    med_bce   = bce_criterion(preds[:,1,:,:], labels[:,1,:,:])
    low_bce   = bce_criterion(preds[:,2,:,:], labels[:,2,:,:])
    high_dice = dice_criterion(preds[:,0,:,:].unsqueeze(1), labels[:,0,:,:].unsqueeze(1))
    med_dice  = dice_criterion(preds[:,1,:,:].unsqueeze(1), labels[:,1,:,:].unsqueeze(1))
    low_dice  = dice_criterion(preds[:,2,:,:].unsqueeze(1), labels[:,2,:,:].unsqueeze(1))
    loss = high_w*(high_bce + high_dice) + med_w*(med_bce + med_dice) + (low_bce + low_dice)
    return loss


def val_model(dataloader, model, bce_criterion, dice_criterion, rank, high_w, med_w):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    high_iou = IoUCalculator('high')
    med_iou = IoUCalculator('medium')
    low_iou = IoUCalculator('low')
    with torch.no_grad():
        for data in dataloader:
            batch_data, batch_labels = data
            batch_data, batch_labels = batch_data.to(rank, dtype=torch.float32, non_blocking=True), batch_labels.to(rank, dtype=torch.float32, non_blocking=True)
            preds = model(batch_data)

            loss = combined_loss(preds, batch_labels, bce_criterion, dice_criterion, high_w, med_w)

            total_loss += loss.item()
            num_batches += 1
            high_iou.update(preds[:,0,:,:], batch_labels[:,0,:,:])
            med_iou.update(preds[:,1,:,:], batch_labels[:,1,:,:])
            low_iou.update(preds[:,2,:,:], batch_labels[:,2,:,:])

    avg_loss = torch.tensor([total_loss / num_batches], device=rank)
    dist.all_reduce(avg_loss)
    avg_loss /= dist.get_world_size()
    if rank == 0:
        print(f"Validation Loss: {avg_loss.item():.6f}", flush=True)
    return avg_loss.item(), high_iou.all_reduce(), med_iou.all_reduce(), low_iou.all_reduce()


def load_model(ckpt_loc, use_ckpt, use_recent, rank, cfg, exp_num, n_epochs):
    arch = cfg['architecture']
    encoder = cfg['encoder']
    lr = cfg['lr']
    weight_decay = cfg.get('weight_decay', 1e-5)
    encoder_lr_mult = cfg.get('encoder_lr_multiplier', 0.1)

    model = smp.create_model(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=3
    )
    model = model.to(rank)

    if rank == 0:
        s = summary(model, input_size=(8,3,256,256), verbose=0)
        print(s)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr * encoder_lr_mult, "weight_decay": weight_decay},
        {"params": model.decoder.parameters(), "lr": lr,                   "weight_decay": weight_decay},
    ], lr=lr)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #    optimizer, T_0=10, T_mult=2
    #)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    start_epoch = 0
    best_loss = 0
    ckpt_pth = None

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if use_ckpt:
        if use_recent:
            ckpt_list = glob.glob('{}{}_{}_exp{}_*.pth'.format(ckpt_loc, arch, encoder, exp_num))
            ckpt_list.sort()
            if ckpt_list:
                most_recent = ckpt_list.pop()
                ckpt_pth = most_recent
        else:
            ckpt_pth = ckpt_loc
        if ckpt_pth:
            if rank == 0:
                print('using this checkpoint: ', ckpt_pth)
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(ckpt_pth, map_location=map_location, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']

    return model, optimizer, scheduler, start_epoch, best_loss


def train_model(train_dataloader, model, bce_criterion, dice_criterion, optimizer, rank, high_w, med_w):
    total_loss = 0.0
    model.train()
    start = time.time()
    num_batches = 0
    for data in train_dataloader:
        optimizer.zero_grad()
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(rank, dtype=torch.float32, non_blocking=True), batch_labels.to(rank, dtype=torch.float32, non_blocking=True)

        preds = model(batch_data)
        loss = combined_loss(preds, batch_labels, bce_criterion, dice_criterion, high_w, med_w)

        total_loss += loss.item()
        num_batches += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    avg_loss = torch.tensor([total_loss / num_batches], device=rank)
    dist.all_reduce(avg_loss)
    avg_loss /= dist.get_world_size()

    if rank == 0:
        print('training time: ', np.round(time.time() - start, 2), flush=True)
        print(f"training loss: {avg_loss.item():.6f}", flush=True)

    return


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
    else:
        data_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = SmokeDataset(data_dict[cat], transform=data_transforms)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=sampler)
    return dataloader


def set_seed(rank):
    seed = 0
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    cudnn.deterministic = True
    cudnn.benchmark = False
    return


def main(rank, world_size, config_fn):
    set_seed(rank)
    torch.manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = False

    exp_num = config_fn.split('exp')[-1].split('.json')[0]

    with open(config_fn) as fn:
        cfg = json.load(fn)

    arch             = cfg['architecture']
    encoder          = cfg['encoder']
    lr               = cfg['lr']
    batch_size       = int(cfg['batch_size'])
    num_workers      = int(cfg['num_workers'])
    encoder_weights  = cfg['encoder_weights']
    n_epochs         = int(cfg.get('n_epochs', 30))
    weight_decay     = cfg.get('weight_decay', 1e-5)
    encoder_lr_mult  = cfg.get('encoder_lr_multiplier', 0.1)
    high_w           = cfg.get('loss_high_weight', 3)
    med_w            = cfg.get('loss_med_weight', 2)

    data_fn = cfg['datapointer']
    with open(data_fn, 'rb') as handle:
        data_dict = pickle.load(handle)

    setup(rank, world_size)

    if rank == 0:
        print('data dict:              ', data_fn)
        print('config fn:              ', config_fn)
        print('number of train samples:', len(data_dict['train']['truth']))
        print('number of val samples:  ', len(data_dict['val']['truth']))
        print('number of test samples: ', len(data_dict['test']['truth']))
        print('learning rate:          ', lr)
        print('batch_size:             ', batch_size)
        print('arch:                   ', arch)
        print('encoder:                ', encoder)
        print('num workers:            ', num_workers)
        print('num gpus:               ', world_size)
        print('n_epochs:               ', n_epochs)
        print('weight_decay:           ', weight_decay)
        print('encoder_lr_mult:        ', encoder_lr_mult)
        print('loss_high_weight:       ', high_w)
        print('loss_med_weight:        ', med_w)

    use_ckpt = False
    use_recent = False
    ckpt_save_loc = cfg['ckpt_save_loc']
    ckpt_loc = None
    if use_ckpt:
        if use_recent:
            ckpt_loc = ckpt_save_loc
        else:
            ckpt_loc = cfg['ckpt']

    model, optimizer, scheduler, start_epoch, best_loss = load_model(ckpt_loc, use_ckpt, use_recent, rank, cfg, exp_num, n_epochs)

    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = smp.losses.DiceLoss(mode='binary')

    prev_iou = 0

    train_loader = prepare_dataloader(rank, world_size, data_dict, 'train', batch_size=batch_size, num_workers=num_workers, train_aug=cfg['train_augmentations'])
    val_loader   = prepare_dataloader(rank, world_size, data_dict, 'val',   batch_size=batch_size, is_train=False, num_workers=num_workers)

    for epoch in range(start_epoch, n_epochs):

        if rank == 0:
            print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)
            start = time.time()

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        train_model(train_loader, model, bce_criterion, dice_criterion, optimizer, rank, high_w, med_w)
        val_loss, high_iou, med_iou, low_iou = val_model(val_loader, model, bce_criterion, dice_criterion, rank, high_w, med_w)

        if rank == 0:
            print("time to run epoch:", np.round(time.time() - start, 2))
            iou, high_iou_val = get_iou(high_iou, med_iou, low_iou)

            log_results(
                os.path.join(cfg['log_dir'], f'stage{cfg["stage"]}_results.csv'),
                {
                    "exp_num":           exp_num,
                    "arch":              arch,
                    "encoder":           encoder,
                    "lr":                lr,
                    "batch_size":        batch_size,
                    "weight_decay":      weight_decay,
                    "encoder_lr_mult":   encoder_lr_mult,
                    "loss_high_weight":  high_w,
                    "loss_med_weight":   med_w,
                    "augmentations":     str(cfg["train_augmentations"]),
                    "epoch":             epoch,
                    "val_loss":          round(val_loss, 6),
                    "val_iou":           round(float(iou), 4),
                    "val_high_iou":      round(float(high_iou_val), 4),
                    "decoder_lr":        round(optimizer.param_groups[1]['lr'], 8),
                }
            )

            if iou > prev_iou and iou > .60 and high_iou_val > .40:
                checkpoint = {
                    'epoch':                epoch + 1,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss':                 val_loss,
                    'iou':                  iou,
                }
                ckpt_pth = '{}{}_{}_exp{}_{}.pth'.format(ckpt_save_loc, arch, encoder, exp_num, int(time.time()))
                torch.save(checkpoint, ckpt_pth)
                print('SAVING MODEL:\n', ckpt_pth, flush=True)
                prev_iou = iou

        scheduler.step(epoch)

    dist.destroy_process_group()


if __name__ == '__main__':

    if len(sys.argv) < 3:
        world_size = 2
    else:
        world_size = int(sys.argv[2])

    if len(sys.argv) < 2:
        print('\n YOU DIDNT SPECIFY CONFIG FILE! ', flush=True)
        sys.exit(1)
    #try:
    config_fn = str(sys.argv[1])
    mp.spawn(main, args=(world_size, config_fn), nprocs=world_size, join=True)

    #except:
        #print("Run failed removing config file")
        #os.remove(config_fn)
