import torch
from torch import nn
from model_o import DeepBrain


def load_checkpoint(path, train_on_gpu, multi_gpu):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """


    # Load in checkpoint
    if train_on_gpu:
        checkpoint = torch.load(path, map_location="cuda")
    else:
        checkpoint = torch.load(path, map_location="cpu")


    model = DeepBrain()

    n_inputs = 64
    #model.post_conv = nn.Conv3d(128, n_inputs, kernel_size=(5, 6, 5)) # changed kernel size to 5 x 6 x 5 from original 5 x 6 x 6 to accomodate the change in input size
    model.classifier = checkpoint['classifier']


        # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[0].requires_grad_(True)
    model.classifier[2] = nn.Dropout(p=0.4)
    model.classifier[3].requires_grad_(True)
    model.post_conv.requires_grad_(True)
    


    # Load in the state dict

    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    if train_on_gpu:
        model = model.to('cuda')
    else:
        model = model.to('cpu')
    
    if multi_gpu:
        model = nn.DataParallel(model)

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer




def save_checkpoint(model, multi_gpu, path):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """



    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary

    if multi_gpu == True:
        checkpoint['classifier'] = model.module.classifier
        checkpoint['state_dict'] = model.module.state_dict()
    else:
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)