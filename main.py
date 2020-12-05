"""
Train the model

Usage:

    $ train.py --model model_name --data_dir /path/to/dataset --run_dir /path/to/experiment

    optional commandline arguments:
    --restore_file /path/to/pretrained/weights
    --run_mode default='test'

Note:
    - for simplicity, all variables pertaining to training (data, labels, loss, metrics, etc.)
      in this file are all torch.tensor objects

"""

## this is a test

# python
import argparse
import os
import logging
import numpy
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from common.data_loader import fetch_dataloader, fetch_subset_dataloader
from common.utils import Params, load_checkpoint, save_checkpoint, set_logger, print_net_summary
from common.objectives import loss_fn, metrics
from common.evaluate import evaluate
from common.train import train
from common.plots import save_batch_summary
from model.build_models import get_network_builder


# commandline
parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', default='./experiments/12-04-2020/resnet152_ImageNet_SGD_MultiStepLR_1',
                    help='Directory containing the runset.json')
parser.add_argument('--data_dir', default='./data/',
                    help='Directory containing the dataset')
parser.add_argument('--restore_file', default='best',
                    help='Optional, name of file in --run_dir containing weights / \
                        hyperparameters to be loaded before training')
parser.add_argument('--run_mode', default='train', help='test mode run a subset \
    of batches to test flow')



def train_and_evaluate(model, optimizer, train_loader, val_loader, loss_fn, metrics, params,
                       run_dir, device, scheduler=None, restore_file=None, writer=None):
    """
    Train the model and evaluate on every epoch

    Args:
        model: (inherits torch.nn.Module) the custom neural network model
        optimizer: (inherits torch.optim) optimizer to update the model parameters
        train_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches
                      the training set
        val_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches
                    the validation set
        loss_fn : (function) a function that takes batch_output (tensor) and batch_labels
                  (np.ndarray) and return the loss (tensor) over the batch
        metrics: (dict) a dictionary of functions that compute a metric using the
                 batch_output and batch_labels
        params: (Params) hyperparameters
        run_dir: (string) directory containing params.json, learned weights, and logs
        restore_file: (string) optional = name of file to restore training from -> no
                      filename extension .pth or .pth.tar/gz
        writer: (tensorboard) tensorboard summary writer
        device: (str) device type; usually 'cuda:0' or 'cpu'

    """

    # reload the weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(run_dir, restore_file + '.pth.zip')
        if os.path.exists(restore_path):
            logging.info("Restoring weights from {}".format(restore_path))
            load_checkpoint(restore_path, model, optimizer)

    best_val_accu = 0.0

    for epoch in range(params.num_epochs):

        # running one epoch
        logging.info("Epoch {} / {}".format(epoch+1, params.num_epochs))

        # logging current learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            logging.info("learning rate = {} for parameter group {}".format(param_group['lr'], i))

        # train for one full pass over the training set
        train_metrics, batch_summ = train(model, optimizer, loss_fn, train_loader, \
            metrics, params, epoch, device, writer)

        # evaluate for one epoch on the validation set
        val_metrics = evaluate(model, loss_fn, val_loader, metrics, params, device)

        # schedule learning rate
        if scheduler is not None:
            scheduler.step()

        # check if current epoch has best accuracy
        val_accu = val_metrics['accuracy']
        is_best = val_accu >= best_val_accu

        # save weights
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
            },
            is_best=is_best,
            checkpoint=run_dir
        )

        # save batch summaries
        save_batch_summary(run_dir, batch_summ)

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
                        writer.add_histogram('layer{}.weight.grad'.format(idx), \
                            m.weight.grad, epoch)


if __name__ == '__main__':

    ### -------- logistics --------###
    # load the params from json file
    args = parser.parse_args()
    json_path = os.path.join(args.run_dir, 'runset.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # parse run parameters
    try:
        # model kwargs could be None
        model_kwargs = params.model['kwargs']
    except KeyError:
        raise "Model keyword argument is None!"
    dataset = params.data['dataset']
    trainloader_kwargs = params.data['trainloader-kwargs']
    trainset_kwargs = params.data['trainset-kwargs']
    valloader_kwargs = params.data['valloader-kwargs']
    valset_kwargs = params.data['valset-kwargs']
    optim_type = params.optimizer['type']
    optim_kwargs = params.optimizer['kwargs']
    lr_type = params.scheduler['type']
    lr_kwargs = params.scheduler['kwargs']

    # tensorboard
    writer = SummaryWriter(args.run_dir)

    # set the logger
    set_logger(os.path.join(args.run_dir, 'train.log'))

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed for reproducible experiments
    torch.manual_seed(200)
    if params.cuda:
        torch.cuda.manual_seed(200)

    ### ------ instantiations ----- ###
    # build model
    model = get_network_builder(params.model['network'])(**model_kwargs)

    # build the optimizer
    optimizer = getattr(optim, optim_type)(model.parameters(), **optim_kwargs)

    # build learning rate scheduler
    scheduler = getattr(optim.lr_scheduler, lr_type)(optimizer, **lr_kwargs)

    ### ------ load dataset ----- ###
    logging.info('Loading dataset...')

    # fetch the data loaders
    # if in test mode, fetch 10 batches (default batch size = 32)
    if args.run_mode == 'test':
        data_loaders = fetch_subset_dataloader(['train', 'val'], args.data_dir, dataset, \
            trainloader_kwargs, trainset_kwargs, valloader_kwargs, valset_kwargs)
        train_dl = data_loaders['train']
        val_dl = data_loaders['val']
    else:
        data_loaders = fetch_dataloader(['train', 'val'], args.data_dir, dataset, \
            trainloader_kwargs, trainset_kwargs, valloader_kwargs, valset_kwargs)
        train_dl = data_loaders['train']
        val_dl = data_loaders['val']

    logging.info('- done.')

    # get a single input batch, add model architecture to tensorboard & log
    # images, labels = next(iter(val_dl))
    # images = images.to(device)
    # 1) write architecutre to tensorboard
    # writer.add_graph(model, images.float())
    # 2) write architecture to log file using torchsummary package
    # print_net_summary(args.run_dir+'/net_summary.log', model, images)

    # move model to GPU
    model = model.to(device)

    # start training
    logging.info('Starting training for {} epoch(s)...'.format(params.num_epochs))

    train_and_evaluate(model, optimizer, train_dl, val_dl, loss_fn, metrics, params,
                       args.run_dir, device, scheduler, args.restore_file, writer)
