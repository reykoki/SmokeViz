from collections import OrderedDict
import torch

#$ckpt_pth = '../deep_learning/models/DeepLabV3Plus_exp0_1730498840.pth'
#ckpt_dst = '../pl_derived_ds/models/ckpt.pth'
#ckpt_pth = '/scratch1/RDARCH/rda-ghpcs/Rey.Koki/Ensemble_SmokeViz/models/DeepLabV3Plus.pth'
#ckpt_dst = '../pl_derived_ds/models/DLV3.pth'
ckpt_pth = '../pl_derived_ds/models/ckpt2.pth'
ckpt_dst = '../pl_derived_ds/models/ckpt2.pth'

def remove_module(ckpt):
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    return new_state_dict

ckpt = torch.load(ckpt_pth, map_location=torch.device('cpu'))
ckpt['model_state_dict'] = remove_module(ckpt['model_state_dict'])
torch.save(ckpt, ckpt_dst)
