import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import torch


def accuracy(train_on_gpu, output, target, topk=(1, )):
    """Compute the topk accuracy(s)"""
    if train_on_gpu:
        output = output.to('cuda')
        target = target.to('cuda')

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Find the predicted classes and transpose
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        # For each k, find the percentage of correct
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        print(res)
        return res



def evaluate(model, test_loader, train_on_gpu, n_classes, criterion, topk=(1, )):
    """Measure the performance of a trained PyTorch model

    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category

    """

    classes = []
    losses = []
    # Hold accuracy results
    acc_results = np.zeros((len(test_loader.dataset), len(topk)))
    i = 0

    model.eval()
    with torch.no_grad():

        # Testing loop
        for data, targets in test_loader:

            # Tensors to gpu
            if train_on_gpu:
                data, targets = data.to('cuda'), targets.to('cuda')
            
            data = data.float()

            # Raw model output
            out = model(data)

            # Iterate through each example
            for pred, true in zip(out, targets):
                # Find topk accuracy
                acc_results[i, :] = accuracy(
                    train_on_gpu, pred.unsqueeze(0), true.unsqueeze(0), topk)
                classes.append(model.idx_to_class[true.item()])
                # Calculate the loss
                loss = criterion(pred.view(1, n_classes), true.view(1))
                losses.append(loss.item())
                i += 1

    # Send results to a dataframe and calculate average across classes
    results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    results['class'] = classes
    results['loss'] = losses
    results = results.groupby(classes).mean()

    return results.reset_index().rename(columns={'index': 'class'})

def plt_confusion_matrix(n_classes,dataloader,train_on_gpu,model):

    confusion_matrix = torch.zeros(n_classes, n_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloader):

            if train_on_gpu:
                inputs = inputs.to('cuda')
                classes = classes.to('cuda')
            else:
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

    return confusion_matrix