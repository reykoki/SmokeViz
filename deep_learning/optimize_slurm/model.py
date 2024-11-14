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


#with open('./dataset_pointers/make_list/pseudo_labeled.pkl', 'rb') as handle:
with open('../dataset_pointers/large/large.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

data_transforms = transforms.Compose([transforms.ToTensor()])

train_set = SmokeDataset(data_dict['train'], data_transforms)
val_set = SmokeDataset(data_dict['val'], data_transforms)
test_set = SmokeDataset(data_dict['test'], data_transforms)

print('there are {} training samples in this dataset'.format(len(train_set)))


def train_model(train_dataloader, val_dataloader, model, n_epochs, start_epoch, exp_num, arch, ckpt_loc, best_loss):
    history = dict(train=[], val=[])
    BCE_loss = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(start_epoch, n_epochs):
        start = time.time()
        for data in train_dataloader:
            batch_data, batch_labels = data
        print('time for epoch: ', time.time()-start)
    return


config_fn = str(sys.argv[1])


with open('configs/{}'.format(config_fn)) as fn:
    hyperparams = json.load(fn)
exp_num = '1'

use_ckpt = False
BATCH_SIZE = int(hyperparams["batch_size"])
train_loader = torch.utils.data.DataLoader(dataset=train_set, pin_memory=True, num_workers=8, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, pin_memory=True, num_workers=8, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

n_epochs = 100
start_epoch = 0
arch = hyperparams['architecture']
model = smp.create_model( # create any model architecture just with parameters, without using its class
        arch=arch,
        encoder_name=hyperparams['encoder'],
        encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
        in_channels=6, # model input channels
        classes=3, # model output channels
)
model = model.to(device)
lr = hyperparams['lr']
optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
best_loss = 0
ckpt_loc = './models/'


train_model(train_loader, val_loader, model, n_epochs, start_epoch, exp_num, arch, ckpt_loc, best_loss)

