import pickle
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#data_pointer_path = './dataset_pointers/make_list/pseudo_labeled.pkl'
data_pointer_path = '/projects/mecr8410/SmokeViz_code/deep_learning/dataset_pointers/smokeviz/SmokeViz.pkl'

with open(data_pointer_path, 'rb') as handle:
    data_dict = pickle.load(handle)

data_transforms = transforms.Compose([transforms.ToTensor()])

train_set = SmokeDataset(data_dict['train'], data_transforms)
val_set = SmokeDataset(data_dict['val'], data_transforms)
test_set = SmokeDataset(data_dict['test'], data_transforms)

print('there are {} training samples in this dataset'.format(len(train_set)))

def compute_iou(pred, true, level, iou_dict):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5) * 1
    intersection = (pred + true == 2).sum()
    union = (pred + true >= 1).sum()
    try:
        iou = intersection / union
        iou_dict[level]['int'] += intersection
        iou_dict[level]['union'] += union
        print('{} density smoke gives: {} IoU'.format(level, iou))
        return iou_dict
    except Exception as e:
        print(e)
        print('there was no {} density smoke in this batch'.format(level))
        return iou_dict

def display_iou(iou_dict):
    high_iou = iou_dict['high']['int']/iou_dict['high']['union']
    med_iou = iou_dict['medium']['int']/iou_dict['medium']['union']
    low_iou = iou_dict['low']['int']/iou_dict['low']['union']
    iou = (iou_dict['high']['int'] + iou_dict['medium']['int'] + iou_dict['low']['int'])/(iou_dict['high']['union'] + iou_dict['medium']['union'] + iou_dict['low']['union'])
    print('OVERALL HIGH DENSITY SMOKE GIVES: {} IoU'.format(high_iou))
    print('OVERALL MEDIUM DENSITY SMOKE GIVES: {} IoU'.format(med_iou))
    print('OVERALL LOW DENSITY SMOKE GIVES: {} IoU'.format(low_iou))
    print('OVERALL OVER ALL DENSITY GIVES: {} IoU'.format(iou))

def val_model(dataloader, model, BCE_loss):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    iou_dict= {'high': {'int': 0, 'union':0}, 'medium': {'int': 0, 'union':0}, 'low': {'int': 0, 'union':0}}
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)

        high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
        med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
        low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
        loss = 3*high_loss + 2*med_loss + low_loss
        #loss = high_loss + med_loss + low_loss
        test_loss = loss.item()
        total_loss += test_loss
        iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
        iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
        iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
    display_iou(iou_dict)


    final_loss = total_loss/len(dataloader)
    print("Validation Loss: {}".format(round(final_loss,8)), flush=True)
    return final_loss

def train_model(train_dataloader, val_dataloader, model, n_epochs, start_epoch, exp_num, arch, ckpt_loc, best_loss):
    history = dict(train=[], val=[])
    BCE_loss = nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, n_epochs):
        total_loss = 0.0
        print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)
        model.train()
        torch.set_grad_enabled(True)
        #for batch_data, batch_labels in train_dataloader:
        for data in train_dataloader:
            batch_data, batch_labels = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            #print(torch.isnan(batch_data).any())
            optimizer.zero_grad() # zero the parameter gradients
            preds = model(batch_data)
            high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
            med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
            low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
            loss = 3*high_loss + 2*med_loss + low_loss
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            total_loss += train_loss
        epoch_loss = total_loss/len(train_dataloader)
        print("Training Loss:   {0}".format(round(epoch_loss,8), epoch+1), flush=True)
        val_loss = val_model(val_dataloader, model, BCE_loss)
        history['val'].append(val_loss)
        history['train'].append(epoch_loss)

        print("prev best loss:", best_loss)
        if val_loss < best_loss:
            print("better loss")
            best_loss = val_loss
        checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
                }
        ckpt_pth = '{}{}_exp{}_{}.pth'.format(ckpt_loc, arch, exp_num, int(time.time()))
        torch.save(checkpoint, ckpt_pth)
        print('SAVING MODEL:\n', ckpt_pth, flush=True)
    print(history)
    return model, history

if len(sys.argv) < 2:
    print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
if len(sys.argv) > 2:
    test_mode = sys.argv[2]

exp_num = str(sys.argv[1])

with open('configs/exp{}.json'.format(exp_num)) as fn:
    hyperparams = json.load(fn)

#use_ckpt = False
use_ckpt = True
BATCH_SIZE = int(hyperparams["batch_size"])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

n_epochs = 100
start_epoch = 0
arch = hyperparams['architecture']
model = smp.create_model( # create any model architecture just with parameters, without using its class
        arch=arch,
        encoder_name=hyperparams['encoder'],
        encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3, # model input channels
        classes=3, # model output channels
)
model = model.to(device)
lr = hyperparams['lr']
optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
best_loss = 10000.0
ckpt_loc = './models/'

if use_ckpt:
    ckpt_list = glob.glob('{}{}_exp{}_*.pth'.format(ckpt_loc, arch, exp_num))
    ckpt_list.sort()
    if ckpt_list:
        # sorted by time
        most_recent = ckpt_list.pop()
        most_recent = './models/DLV3P_exp1_1719683871.pth'
        most_recent = './models/DeepLabV3Plus_exp1_1726214171.pth'
        print('using this checkpoint: ', most_recent)
        checkpoint=torch.load(most_recent)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
    else:
        use_ckpt = False


train_model(train_loader, val_loader, model, n_epochs, start_epoch, exp_num, arch, ckpt_loc, best_loss)

