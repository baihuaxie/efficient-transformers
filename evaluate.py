""" Evaluate trained model on the validation set """

import logging
import numpy as np

import torch

def evaluate(model, loss_fn, dataloader, metrics, params, device):
    """
    Evalute the model

    Args:
        model: (torch.nn.Module) the network model to be evaluated
        loss_fn: (function) a function which takes output_batch and labels_batch and returns the loss over a batch
        dataloader: (torch.utils.data.DataLoader) a DataLoader object that returns access to the validation set
        metrics: (dict) a dictionary; each element contains a function to return the value for the metric
        params: (Params) hyperparameters
        device: (str) device type; usually 'cuda:0' or 'cpu'

    Returns:
        metrics_mean: (dict) a dictionary containing mean value for each metrics evaluated on the validation set

    """

    # set model to evaluateion mode
    model.eval()

    # initialize summary for current
    summ = []

    # compute metrics over the dataset
    for _, (data_batch, labels_batch) in enumerate(dataloader):

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.to(device,
                non_blocking=True), labels_batch.to(device, non_blocking=True)

        # compute the model output
        with torch.no_grad():
            output_batch = model(data_batch)

        # compute loss
        loss = loss_fn(output_batch, labels_batch)
        loss_detach = loss.detach()

        # move data to cpu
        # compute metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch.to('cpu'), labels_batch.to('cpu')) for metric in metrics.keys()}
        summary_batch['loss'] = loss_detach.item()
        summ.append(summary_batch)

    # compute the mean of all metrics on validation set
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0].keys()}
    metrics_string = ' ; '.join('{}: {:05.3f}'.format(k, v) for k, v in metrics_mean.items())

    logging.info("- Eval metrics: {}".format(metrics_string))

    return metrics_mean
    