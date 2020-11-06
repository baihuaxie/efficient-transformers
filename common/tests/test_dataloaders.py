"""
    test dataloader functions
"""

import pytest
import os.path as op

from common.utils import Params
from common.data_loader import fetch_dataloader, \
    select_n_random, fetch_subset_dataloader

@pytest.fixture
def init_params():
    json_path = op.join('../', 'runset.json')
    params = Params(json_path)
    dataset = params.data['dataset']
    data_dir = '../../data/'
    dataset_kwargs = params.data['kwargs']
    return dataset, data_dir, dataset_kwargs

def test_fetch_dataloaders(init_params):
    """
    """
    dataset, data_dir, dataset_kwargs = init_params
    dataloaders = fetch_dataloader(['train'], data_dir, \
        dataset, **dataset_kwargs)
    train_dl = dataloaders['train']


def test_select_n_random(init_params):
    """
    """
    num = 10
    dataset, data_dir, dataset_kwargs = init_params
    images, labels = select_n_random('train', data_dir, \
        dataset, n=num)
    assert images.shape[0] == num
    assert labels.shape[0] == num


def test_fetch_subset_dataloaders(init_params):
    """
    """
    dataset, data_dir, dataset_kwargs = init_params
    dataloaders = fetch_subset_dataloader(['train'], \
        data_dir, dataset, 32, **dataset_kwargs)
