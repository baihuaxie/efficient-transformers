"""
    Train one epoch
"""

import logging
from tqdm import tqdm
import numpy as np

from common import utils

def train(model, optimizer, loss_fn, dataloader, metrics, params, epoch, device, writer=None):
    """
    Train the model on num_steps batches

    Args:
        model: (torch.nn.Module) model to be trained
        optimizer: (torch.optim) optimizer to update weights
        loss_fn: (function) a function that takes batch_output and batch_labels to
                 return the loss over the batch
        dataloader: (torch.utils.data.DataLoader) a DataLoader object to facilitate
                 accessing data from the training set
        metrics: (dict) contains functions to return the value of each metric; metrics
                 functions accept torch.tensor inputs
        params: (Params) hyperparameters
        device: (str) device type; usually 'cuda:0' or 'cpu'

    """

    # set the model to train mode
    model.train()

    # initialize summary for current training loop
    # summ -> a list containing for each element a dictionary object, which stores
    # metric-value pairs obtained during training
    summ = []

    # initialize a running average object for loss
    loss_avg = utils.RunningAverage()

    # use tqdm for progress bar during training
    with tqdm(total=len(dataloader)) as prog:

        # standard way to access DataLoader object for iteration over dataset
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            # move to GPU if available
            train_batch, labels_batch = train_batch.to(device,
                non_blocking=True), labels_batch.to(device, non_blocking=True)

            # compute model output
            output_batch = model(train_batch)

            # compute loss
            loss = loss_fn(output_batch, labels_batch)
            loss_detach = loss.detach()

            # clear previous gradients, back-propagate gradients of loss w.r.t. all parameters
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # evaluate training summaries at certain iterations
            if i % params.save_summary_steps == 0:
                # move data to cpu
                # train data and labels are torch.tensor objects
                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch.to('cpu'), \
                                labels_batch.to('cpu')) for metric in metrics.keys()}
                # add 'loss' as a metric -> because loss is already computed by loss_fn,
                # no need to define another metric function
                summary_batch['loss'] = loss_detach.item()

                # write training summary to tensorboard if applicable
                if writer is not None:
                    for metric, value in summary_batch.items():
                        writer.add_scalar(
                            'training '+metric, value, epoch*len(dataloader)+i
                        )

                # append summary
                summ.append(summary_batch)

            # update the running average loss
            loss_avg.update(loss_detach.item())

            # update progress bar to show running average for loss
            prog.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            prog.update()

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0].keys()}
        metrics_string = ' ; '.join('{}: {:5.03f}'.format(k, v) for k, v in metrics_mean.items())

        logging.info("- Train metrics: {}".format(metrics_string))

    return metrics_mean
