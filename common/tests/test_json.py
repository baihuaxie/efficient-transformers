"""
    test runset.json file
"""

import os.path as op
import pytest
from importlib import import_module

import torch.optim as optim

from common.utils import Params

from model.build_models import get_network_builder
from common.data_loader import fetch_dataloader

@pytest.fixture
def init_runset():
    """
    initialize from runset.json
    """
    json_path = op.join('../', 'runset.json')
    params = Params(json_path)
    return params

@pytest.fixture
def init_params():
    """
    """
    json_path = op.join('../', 'parameters.json')
    return Params(json_path)

@pytest.fixture
def init_model():
    """
    return a network model
    """
    return get_network_builder('resnet18')()


def test_params(init_params):
    """
    """
    params = init_params
    print(params.optimizer[1])
    print(len(params.optimizer))


def test_customize_optimzier(init_params, init_model):
    """
    test customize optimizers
    """
    params = init_params
    model = init_model
    print("")
    for idx in range(len(params.optimizer)):
        optim_params = params.optimizer[idx]
        kwargs = optim_params['kwargs']
        optimizer = getattr(optim, optim_params['type'])( \
            model.parameters(), **kwargs)
        #print(optimizer)


def test_customize_scheduler(init_params, init_model):
    """
    test customize lr scheduler
    """
    params = init_params
    model = init_model
    print("")
    optim_params = params.optimizer[0]
    optim_kwargs = optim_params['kwargs']
    optimizer = getattr(optim, optim_params['type'])( \
        model.parameters(), **optim_kwargs)

    for idx in range(len(params.scheduler)):
        scheduler_params = params.scheduler[idx]
        scheduler_kwargs = scheduler_params['kwargs']
        scheduler = getattr(optim.lr_scheduler, scheduler_params['type'])( \
            optimizer, **scheduler_kwargs)
        #print(scheduler)


def test_customize_datasets(init_params):
    """
    """


def test_runset(init_runset, init_model):
    """
    test using runset.json to launch a single experiment
    """
    params = init_runset
    #model = init_model
    print("")

    # model
    model = get_network_builder(params.model)()
    print(model)

    # dataset
    dataset = params.data['dataset']
    data_dir = '../../data/'
    dataset_kwargs = params.data['kwargs']
    dataloaders = fetch_dataloader(['train', 'test'], data_dir, \
        dataset, **dataset_kwargs)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']
    print(train_dl)

    # optimizer
    optim_kwargs = params.optimizer['kwargs']
    optimizer = getattr(optim, params.optimizer['type'])( \
        model.parameters(), **optim_kwargs)
    print(optimizer)

    # scheduler
    scheduler_kwargs = params.scheduler['kwargs']
    scheduler = getattr(optim.lr_scheduler, params.scheduler['type'])( \
        optimizer, **scheduler_kwargs)
    print(scheduler)


