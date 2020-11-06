"""
Train the model

Usage:

    $ train.py --model model_name --data_dir /path/to/dataset --exp_dir /path/to/experiment

    optional commandline arguments:
    --restore_file /path/to/pretrained/weights
    --run_mode default='test'

Note:
    - for simplicity, all variables pertaining to training (data, labels, loss, metrics, etc.) in this file are all torch.tensor objects

"""

# python
import argparse
import os
import logging
from importlib import import_module
import numpy as np
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# models
import model.resnet as resnet
import model.densenet as densenet
import model.mobilenetv1 as mobilenetv1

# utilities
import data_loader
import utils
import objectives as obj
from evaluate import evaluate


# commandline
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet18',
                    help='Specify ResNet variant to be trained')
parser.add_argument('--data_dir', default='./data/',
                    help='Directory containing the dataset')
parser.add_argument('--exp_dir', default='./experiments/launch-test',
                    help='Directory containing the params.json')
parser.add_argument('--restore_file', default='best',
                    help='Optional, name of file in --exp_dir containing weights / hyperparameters to be loaded before training')
parser.add_argument('--run_mode', default='test', help='test mode run a subset of batches to test flow')


def train(model, optimizer, loss_fn, dataloader, metrics, params, epoch, device, writer=None):
    """
    Train the model on num_steps batches

    Args:
        model: (torch.nn.Module) model to be trained
        optimizer: (torch.optim) optimizer to update weights
        loss_fn: (function) a function that takes batch_output and batch_labels to return the loss over the batch
        dataloader: (torch.utils.data.DataLoader) a DataLoader object to facilitate accessing data from the training set
        metrics: (dict) contains functions to return the value of each metric; metrics functions accept torch.tensor inputs
        params: (Params) hyperparameters
        device: (str) device type; usually 'cuda:0' or 'cpu'

    """

    # set the model to train mode
    model.train()

    # initialize summary for current training loop
    # summ -> a list containing for each element a dictionary object, which stores metric-value pairs obtained during training
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
                summary_batch = {metric: metrics[metric](output_batch.to('cpu'), labels_batch.to('cpu'))
                                 for metric in metrics.keys()}
                # add 'loss' as a metric -> because loss is already computed by loss_fn, no need to define another metric function
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



def train_and_evaluate(model, optimizer, train_loader, val_loader, loss_fn, metrics, params,
                       exp_dir, device, scheduler=None, restore_file=None, writer=None):
    """
    Train the model and evaluate on every epoch

    Args:
        model: (inherits torch.nn.Module) the custom neural network model
        optimizer: (inherits torch.optim) optimizer to update the model parameters
        train_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches the training set
        val_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches the validation set
        loss_fn : (function) a function that takes batch_output (tensor) and batch_labels (np.ndarray) and return the loss (tensor) over the batch
        metrics: (dict) a dictionary of functions that compute a metric using the batch_output and batch_labels
        params: (Params) hyperparameters
        exp_dir: (string) directory containing params.json, learned weights, and logs
        restore_file: (string) optional = name of file to restore training from -> no filename extension .pth or .pth.tar/gz
        writer: (tensorboard) tensorboard summary writer
        device: (str) device type; usually 'cuda:0' or 'cpu'

    """

    # reload the weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(exp_dir, restore_file + '.pth.zip')
        if os.path.exists(restore_path):
            logging.info("Restoring weights from {}".format(restore_path))
            utils.load_checkpoint(restore_path, model, optimizer)

    best_val_accu = 0.0

    for epoch in range(params.num_epochs):

        # running one epoch
        logging.info("Epoch {} / {}".format(epoch+1, params.num_epochs))

        # logging current learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            logging.info("learning rate = {} for parameter group {}".format(param_group['lr'], i))

        # train for one full pass over the training set
        train_metrics = train(model, optimizer, loss_fn, train_loader, metrics, params, epoch, device, writer)

        # evaluate for one epoch on the validation set
        val_metrics = evaluate(model, loss_fn, val_loader, metrics, params, device)

        # schedule learning rate
        if scheduler is not None:
            scheduler.step()

        # check if current epoch has best accuracy
        val_accu = val_metrics['accuracy']
        is_best = val_accu >= best_val_accu

        # save weights
        utils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
            },
            is_best=is_best,
            checkpoint=exp_dir
        )

        # if best accuray
        if is_best:
            logging.info("- Found new best accuray model at epoch {}".format(epoch+1))
            best_val_accu = val_accu

        # add training log to tensorboard
        if writer is not None:

            # train and validation per-epoch mean metrics
            for metric, value in train_metrics.items():
                if metric in val_metrics.keys():
                    writer.add_scalars(metric, {'train': value, 'val': val_metrics[metric]}, epoch)

            # layer weights / gradients distributions
            for idx, m in enumerate(model.modules()):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if m.weight is not None:
                        writer.add_histogram('layer{}.weight'.format(idx), m.weight, epoch)
                    if m.weight.grad is not None:
                        writer.add_histogram('layer{}.weight.grad'.format(idx), m.weight.grad, epoch)


if __name__ == '__main__':

    ### -------- logistics --------###
    # load the params from json file
    args = parser.parse_args()
    json_path = os.path.join(args.exp_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    myParams = utils.Params(json_path)

    # load the models from json file
    models_dict = utils.Params('./model/models.json')

    # tensorboard
    writer = SummaryWriter(args.exp_dir)

    # set the logger
    utils.set_logger(os.path.join(args.exp_dir, 'train.log'))

    # use GPU if available
    myParams.cuda = torch.cuda.is_available()
    myDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed for reproducible experiments
    torch.manual_seed(200)
    if myParams.cuda:
        torch.cuda.manual_seed(200)

    ### --------- model ---------- ###
    # define the model
    net, model = models_dict.dict[args.model].split('.')
    myModel = getattr(import_module('.'+net, 'model'), model)().to(myDevice)

    # add model architecture to tensorboard & log
    images, labels = data_loader.select_n_random('train', args.data_dir, n=2)
    images, labels = images.to(myDevice), labels.to(myDevice)
    # write to tensorboard
    writer.add_graph(myModel, images.float())
    # write to log file
    utils.print_net_summary(args.exp_dir+'/net_summary.log', myModel, images)


    ### ------ data pipeline ----- ###
    logging.info('Loading datasets...')

    # fetch the data loaders
    # if in test mode, fetch 10 batches
    if args.run_mode == 'test':
        data_loaders = data_loader.fetch_subset_dataloader(
            ['train', 'test'], args.data_dir, myParams, 10)
        train_dl = data_loaders['train']
        test_dl = data_loaders['test']
    else:
        data_loaders = data_loader.fetch_dataloader(
            ['train', 'test'], args.data_dir, myParams)
        train_dl = data_loaders['train']
        test_dl = data_loaders['test']

    logging.info('- done.')

    ### ------ optimizer --------- ###
    # define the optimizer
    if myParams.optimizer == 'Adam':
        # use Adam
        myOptimizer = optim.Adam(myModel.parameters(), lr=myParams.initial_lr,
                                 weight_decay=myParams.weight_decay)
    if myParams.optimizer == 'SGD':
        # use SGD w.t. Nesterov momentum
        myOptimizer = optim.SGD(myModel.parameters(), lr=myParams.initial_lr, momentum=myParams.momentum,
                                weight_decay=myParams.weight_decay, nesterov=True)

    if myParams.optimizer == 'RMSprop':
        # use RMSprop
        myOptimizer = optim.RMSprop(myModel.parameters(), lr=myParams.initial_lr, momentum=myParams.momentum,
                                    weight_decay=myParams.weight_decay, alpha=myParams.alpha)

    ### ------ scheduler --------- ###
    # define learning rate scheduler
    if myParams.scheduler == 'MultiStepLR':
        myScheduler = optim.lr_scheduler.MultiStepLR(myOptimizer, milestones=myParams.scheduler_milestones,
                                                     gamma=myParams.scheduler_gamma)

    if myParams.scheduler == 'StepLR':
        myScheduler = optim.lr_scheduler.StepLR(myOptimizer, step_size=myParams.scheduler_step_size,
                                                gamma=myParams.scheduler_gamma)

    if myParams.scheduler == 'OneCycleLR':
        # the one cycle scheduler
        # warm up lr from initial_lr = max_lr / div_factor to max_lr for first div_factor steps
        # then decay to min_lr = initial_lr / final_div_factor by cosine or linear annealing
        # default: pct_start=5/total_steps, steps_per_epoch=1, final_div_factor=1e4
        myScheduler = optim.lr_scheduler.OneCycleLR(myOptimizer, max_lr=myParams.scheduler_max_lr, epochs=myParams.num_epochs,
                                                    steps_per_epoch=1, div_factor=myParams.scheduler_div_factor,
                                                    pct_start=5/myParams.num_epochs,
                                                    final_div_factor=myParams.scheduler_final_div_factor)

    # fetch loss function and metrics
    my_loss_fn = obj.loss_fn
    my_metrics = obj.metrics

    ### ----- train the model ----- ###
    logging.info('Starting training for {} epoch(s)...'.format(myParams.num_epochs))

    train_and_evaluate(myModel, myOptimizer, train_dl, test_dl, my_loss_fn, my_metrics, myParams,
                       args.exp_dir, myDevice, myScheduler, args.restore_file, writer)
