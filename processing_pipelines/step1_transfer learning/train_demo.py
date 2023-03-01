# import packages and functions

import os
import sys
import glob
import numpy as np
import pandas as pd

import torch
from torch import optim, cuda
from torch import nn
from torch.functional import F
torch.backends.cudnn.benchmark = False

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


from torchsummary import summary
from timeit import default_timer as timer

from model_o import DeepBrain
from dataset import BrainDataset
from torch.utils.data import DataLoader
from get_pretrained_model import get_pretrained_model
from train import train
from guided_backprop_o import GuidedBackprop
from load_save_checkpoint import load_checkpoint, save_checkpoint
from test import accuracy, evaluate

## Visualizations
import matplotlib.pyplot as plt

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DeepBrain()
criterion = nn.CrossEntropyLoss()

for param in model.parameters():
    param.requires_grad = False

n_inputs = 64
model.post_conv = nn.Conv3d(128, n_inputs, kernel_size=(5, 6, 5)) 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

h = 81
data = torch.randn(1, 27, h, h, h)

# Clear gradients
optimizer.zero_grad()
# Predicted outputs are log probabilities
output = model(data)

target = torch.randint(0, 7, (1,))
# Loss and backpropagation of gradients
loss = criterion(output, target)
loss.backward()

print(sum([p.grad.abs().sum() for p in model.parameters() if p.requires_grad==True]))
# tensor(1184.2426)

w0 = model.post_conv.weight.clone()

# Update the parameters
optimizer.step()

w1 = model.post_conv.weight.clone()

print((w1 - w0).abs().sum())
# tensor(1190.6450, grad_fn=<SumBackward0>)