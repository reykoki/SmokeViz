import torch
from torch.utils.data import Dataset
import tifffile


class SmokeDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_fns = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_fns['data'])

    def __getitem__(self, idx):
        data_fn = self.data_fns['data'][idx]
        truth_fn = self.data_fns['truth'][idx]
        data_img = tifffile.imread(data_fn)
        truth_img = tifffile.imread(truth_fn)
        data_tensor = self.transform(data_img)#.unsqueeze_(0)
        #data_tensor = torch.clip(data_tensor, min=0, max=1.25)
        truth_tensor = self.transform(truth_img)#.unsqueeze_(0)
        truth_tensor = (truth_tensor > 0.0) * 1.0
        truth_tensor = truth_tensor.type(torch.float32)

        return data_tensor, truth_tensor
