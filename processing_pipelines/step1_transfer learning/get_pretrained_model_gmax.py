def get_pretrained_model_gmax(n_classes,drop_out,train_on_gpu,multi_gpu):


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
    from model_o import DeepBrain
    from model_gmaxpool import DeepBrainGMax

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    model = DeepBrainGMax()
    if train_on_gpu:
        model.load_state_dict(torch.load('./checkpoint/checkpoint_o.pth.tar',map_location="cuda")['state_dict'], strict=False)
    else:
        model.load_state_dict(torch.load('./checkpoint/checkpoint_o.pth.tar',map_location="cpu")['state_dict'], strict=False)

    #model.n_classes = n_classes
    # freeze the parameters
    
    #for param in model.parameters():
        #param.requires_grad = False
    
   

    #if drop_out:
        #model.classifier[2].requires_grad_(True)
        #model.classifier[5].requires_grad_(True)
        
    #else:
        #model.classifier[1].requires_grad_(True)
        #model.classifier[3].requires_grad_(True)

    
    # reinitialize the weights of the fully connected layers and the last conv layer

    

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