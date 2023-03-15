def get_pretrained_model(n_classes,drop_out,train_on_gpu,multi_gpu):


    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """

    import os
    import sys
    import glob
    import numpy as np
    import pandas as pd

    import torch
    from torch import nn
    #from model_o import DeepBrain
    #from model_gmaxpool import DeepBrainGMax
    from model_dropoutconv import DeepBrain

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    model = DeepBrain()
    if train_on_gpu:
        model.load_state_dict(torch.load('./checkpoint/checkpoint_o.pth.tar',map_location="cuda")['state_dict'], strict=False)
    else:
        model.load_state_dict(torch.load('./checkpoint/checkpoint_o.pth.tar',map_location="cpu")['state_dict'], strict=False)

    # freeze the parameters
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.post_conv.requires_grad_(True)
    model.classifier[1].requires_grad_(True)

    
    # reinitialize the weights of the fully connected layers and the last conv layer

    n_inputs = 64

    model.n_classes = n_classes
    #model.post_conv = nn.Conv3d(128, n_inputs, kernel_size=(5, 6, 5)) # changed kernel size to 5 x 6 x 5 from original 5 x 6 x 6 to accomodate the change in input size
    
    if drop_out:
        #model.classifier[0] = nn.Linear(n_inputs,n_inputs)
        model.classifier[0] = nn.Dropout(p=0.25)
        model.classifier[3] = nn.Dropout(p=0.2)
        model.classifier[4] = nn.Linear(64, n_classes)

        #model.classifier = nn.Sequential(
                #nn.Linear(n_inputs,n_inputs),
                #nn.ReLU(inplace=True),
                #nn.Dropout(p=0.375),
                #nn.Linear(n_inputs,n_inputs),
                #nn.ReLU(inplace=True),
                #nn.Dropout(p=0.25),
                #nn.Linear(n_inputs, n_classes),
                #nn.LogSoftmax())

    else:
        model.classifier[0] = nn.Linear(n_inputs,n_inputs)
        model.classifier[2] = nn.Linear(64, n_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


    # Move to gpu and parallelize
    
    if train_on_gpu:
        model = model.to('cuda')
    else:
        model = model.to('cpu')


    if multi_gpu:
        model = nn.DataParallel(model)


    return model