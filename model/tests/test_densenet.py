""" test densenet.py """

import sys
import pytest

import torch

sys.path.append('..')

import model.densenet as net
import data_loader as dataloader
import utils


@pytest.fixture
def datadir():
    """ set directory containing dataset """
    return '../data/'

@pytest.fixture
def is_cuda():
    """ if cuda is available """
    return torch.cuda.is_available()


@pytest.fixture
def params():
    """ read params from json file """
    return utils.Params('../experiments/base-model/params.json')

@pytest.fixture
def select_data(datadir, is_cuda):
    """ select n random images + labels from train """
    images, labels = dataloader.select_n_random('train', datadir, n=2)
    if is_cuda:
        images, labels = images.cuda(), labels.cuda()
    return images.float(), labels

def test_densenet40_k12(select_data, is_cuda):
    """ test densenet40_k12 model """
    model = net.densenet40_k12()
    if is_cuda:
        model = model.cuda()
    images, _ = select_data
    output = model(images)
    utils.print_net_summary('./log_test', model, images)


