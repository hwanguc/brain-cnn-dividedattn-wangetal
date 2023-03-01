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
from get_pretrained_model_gmax import get_pretrained_model_gmax
from train import train
from guided_backprop_o import GuidedBackprop
from load_save_checkpoint import load_checkpoint, save_checkpoint
from test import accuracy, evaluate, plt_confusion_matrix

## Visualizations
import matplotlib.pyplot as plt

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# global hyper parameters

## Change to fit hardware
batch_size = 4
n_epochs = 80
learning_rate = 0.0008

## model state and checkpoint

save_file_name = './checkpoint/3dconv-transfer_state_dict_v3.pt'
checkpoint_path = './checkpoint/3dconv-transfer_checkpoint_v2.pth'

## Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

## Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False
else:
    multi_gpu = False


# data curation and loading

## read images and labels

dat_tr = pd.read_csv("./data/tr_img_label_local_5va_2lab.csv")
img_tr = dat_tr.iloc[:,0].values.tolist()
lab_tr = dat_tr.iloc[:,1].values.tolist()

dat_va = pd.read_csv("./data/va_img_label_local_5va_2lab.csv")
img_va = dat_va.iloc[:,0].values.tolist()
lab_va = dat_va.iloc[:,1].values.tolist()


dat_te = pd.read_csv("./data/test_img_label_local_5va_2lab.csv")
img_te = dat_te.iloc[:,0].values.tolist()
lab_te = dat_te.iloc[:,1].values.tolist()

## read to datasets
datasets_tr = BrainDataset(img_tr, lab_tr, is_train=True, is_resize=True) # resize the data for the input image to match the dimension of that in the paper.
datasets_va = BrainDataset(img_va, lab_va, is_train=False, is_resize=True)
datasets_te = BrainDataset(img_te, lab_te, is_train=False, is_resize=True)

## read to dataloader

dataloader_tr = DataLoader(datasets_tr, batch_size=batch_size, shuffle=True)
dataloader_va = DataLoader(datasets_va, batch_size=batch_size, shuffle=True)
dataloader_te = DataLoader(datasets_te, batch_size=batch_size, shuffle=True)


## dataloader iterators

trainiter = iter(dataloader_tr)
features, labels = next(trainiter)
features.shape, labels.shape

## number of classes
#n_classes = 4
n_classes = 2

## class to idx mapping


#class_to_idx = {
    #"SpEasy_VisEasy": 0,
    #"SpEasy_VisHard": 1,
    #"SpHard_VisEasy": 2,
    #"SpHard_VisHard": 3
#}

class_to_idx = {
    "SpEasy": 0,
    "SpHard": 1,
}



# load the pretrained model state

drop_out = True
model = get_pretrained_model(n_classes,drop_out,train_on_gpu,multi_gpu)
#model = get_pretrained_model_gmax(n_classes,drop_out,train_on_gpu,multi_gpu)

if train_on_gpu:
    if multi_gpu:
        summary(
            model.module,
            input_size=(27, 75, 93, 81),   # the input_size needs to be updated! (27, 79, 95, 79) current data, (27, 75, 93, 81) wangetal 2019
            batch_size=batch_size,
            device='cuda')
    else:
        summary(
            model, input_size=(27, 75, 93, 81), batch_size=batch_size, device='cuda')  # the input_size needs to be updated!!!
else:
    summary(
        model, input_size=(27, 75, 93, 81), batch_size=batch_size, device='cpu')  # the input_size needs to be updated!!!
    


## save idx to label mappings to model

model.class_to_idx = class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}
#list(model.idx_to_class.items())[:4]
list(model.idx_to_class.items())[:2]
    

# training and validation

## implementation

### hyper parameters

criterion = nn.NLLLoss()
#criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.0005)
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=5e-6)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

### print the parameters that needs to be updated during training

for p in optimizer.param_groups[0]['params']:
    if p.requires_grad:
        print(p.shape)

for name, param in model.named_parameters():
    print(name, param.requires_grad)

### the ACTUAL training!

model, history = train(
    model,
    train_on_gpu,
    criterion,
    optimizer,
    dataloader_tr,
    dataloader_va,
    save_file_name=save_file_name,
    max_epochs_stop=15,
    n_epochs=n_epochs,
    print_every=1)


## training results

### losses

plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Negative Log Likelihood')
plt.title('Training and Validation Losses')


### accuracy

plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 * history[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy')
plt.title('Training and Validation Accuracy')




# save the transfer learnt model

save_checkpoint(model, multi_gpu, checkpoint_path)

# load a trainsfer learnt model from a previous training session

model, optimizer = load_checkpoint(checkpoint_path, train_on_gpu, multi_gpu)

if train_on_gpu:
    if multi_gpu:
        summary(
            model.module,
            input_size=(27, 79, 95, 79),   # the input_size needs to be updated!!!
            batch_size=batch_size,
            device='cuda')
    else:
        summary(
            model, input_size=(27, 79, 95, 79), batch_size=batch_size, device='cuda')  # the input_size needs to be updated!!!
else:
    summary(
        model, input_size=(27, 79, 95, 79), batch_size=batch_size, device='cpu')  # the input_size needs to be updated!!!


# test

## predict a target (as an example)

testiter = iter(dataloader_te)


### Get a batch of testing images and labels
features, targets = next(testiter)

features = features.float()

### overall accuracy for one batch in the test data iterator 

if train_on_gpu:
    res = accuracy(train_on_gpu,model(features.to('cuda')), targets, topk=(1, ))
else:
    res = accuracy(train_on_gpu,model(features), targets, topk=(1, ))



# Evaluate the model on all the training data

## accuracy for each class

results = evaluate(model, dataloader_te,train_on_gpu,n_classes, criterion, topk=(1, 3))
results.head()

## a confusion matrix

plt_confusion_matrix(n_classes,dataloader_te,train_on_gpu,model)

### code below:

import seaborn as sn

confusion_matrix = torch.zeros(n_classes, n_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloader_te):
        inputs = inputs.to('cpu')
        classes = classes.to('cpu')
        inputs = inputs.float()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


confusion_matrix = confusion_matrix/confusion_matrix.sum(axis=1)[:,None]

labels_all = []
for idx, class_ in model.idx_to_class.items():
    labels_all.append(class_)

df_cm = pd.DataFrame(confusion_matrix, index = [i for i in labels_all],
                     columns = [i for i in labels_all])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig('output.png')


print(confusion_matrix.diag()/confusion_matrix.sum(1))










