import numpy as np
import nibabel as nib
#import torch
#from torch import nn
#from torch.functional import F
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self, file_list, label_list, is_train=True, is_resize=True):
        self.is_train = is_train
        self.is_resize = is_resize
        self.input_shape = 27
        self.file_list = file_list
        self.label_list = label_list
        
    def __getitem__(self, index):
        img = self.normalize_data(nib.load(self.file_list[index]).get_fdata())
        target = self.label_list[index]
        #target = self.label_list[index]
        #target = F.one_hot(torch.tensor(target), num_classes=4)

        return img, target
    
    def __len__(self):
        return len(self.label_list)
    
    def normalize_data(self, data):
        #if self.is_train:
            #select_idx = np.random.randint(low=self.input_shape, high=data.shape[-1], size=1)[0]
            #data = data[8:-8, 8:-8, :-10, select_idx-self.input_shape:select_idx]
        #else:
            #data = data[8:-8, 8:-8, :-10, :self.input_shape]
        data = data[8:-8, 8:-8, :-10, :self.input_shape]
        if self.is_resize:
            data = data[2:-2,1:-1,:,:]
            data = np.pad(data, ((0, 0),(0, 0), (1, 1), (0, 0)))

        
        data = data / np.amax(data)
        data[~ np.isfinite(data)] = 0
        return data.transpose(3, 0, 1, 2)