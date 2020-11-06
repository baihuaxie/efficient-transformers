"""
    Reads raw dataset and returns dataloader(s)

    This file should take the following as inputs:
    - raw dataset -> url for download; file path for local
    - hyper-parameters, e.g., num_workers, pinning memeory, batch_size, etc.

    This file should return:
    - torch.utils.data.DataLoader objects -> conceptually, should distinct dataloaders for train, val, test datasets

    The following items need to be implemented in this file:

    1) a class definition for dataset
    -- if using pytorch built-in datasets, e.g., torchvision.datasets, can directly call the dataset class and omit custom definition
    -- if using custom datasets, need to define a class for that dataset
       - class needs to inherit from either Dataset or IterableDataset pytorch classes
       - for Dataset class, need to define __getitem__() and __len__() methods
       - for IterableDataset class, need to define __iter__() method
    -- class input: directory, tansforms
       - needs to apply transforms to raw dataset before return

    2) a set of transform pipelines
    -- for each dataloader type (train, val, test, etc.), may need to define different transform pipelines
    -- usually uses torchvision.transforms.Compose() to construct the pipelines

    3) a function / method to return the corresponding DataLoader object for each type
    -- need to instantiate the defined or built-in dataset class
    -- usually uses torch.utils.data.DataLoader() to build DataLoader objects
    -- this function / method is then called in main scripts to load dataset

"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

# transforms pipeline for train set
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# transforms pipeline for test set
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

def fetch_dataset(types, datadir, dataset_name=None):
    """
    Fetches the dataset objects for each type from datadir

    Used subsequently to fetch dataloader objects or fetch random samples

    Args:
        types: (list) has one or more of "train, val, test" depending on the dataset
        datadir: (str) file path containing the raw dataset
        dataset_name: (str) name of the dataset to be loaded

    Returns:
        dataset: (dict) contains the dataset object for each type in types
    """

    datasets = {}

    for split in ['train', 'val', 'test']:
        if split in types:

            # load train set
            if split == 'train':
                dataset = CIFAR10(datadir, download=False, train=True, transform=train_transform)

            # load test set
            if split == 'test':
                dataset = CIFAR10(datadir, download=False, train=False, transform=test_transform)

            datasets[split] = dataset

    return datasets


def select_n_random(dataset_type, datadir, n=1):
    """
    Select n random [data, label] points from dataset

    Args:
        dataset_type: (str) a string of either 'train', 'val' or 'test'
        datadir: (str) file path to the raw dataset

    Return:
        n data points + corresponding labels (both tensors)
        data: (tensor) data points stored in ndarrays
        labels: (tensor) labels stored in ndarrays

    """

    dataset = fetch_dataset(dataset_type, datadir)[dataset_type]
    data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)
    labels = torch.Tensor(dataset.targets)

    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]



def fetch_dataloader(types, datadir, params):
    """
    Fetches the dataloader objects for each type from datadir.

    Args:
        types: (list) has one or more of "train, val, test" depending on the dataset
        datadir: (str) file path containing the raw dataset
        params: (Params object) hyperparameters controlling data loading behavior

    Returns:
        dataloadr: (dict) contains the DataLoader objects for each type in types
    """

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:

            # apply train set tranforms if train data
            if split == 'train':
                # trainset = CIFAR10(datadir, download=False, train=True, transform=train_transform)
                trainset = fetch_dataset(split, datadir, 'CIFAR10')['train']
                dataloader = DataLoader(trainset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.pin_cuda)

            # apply test set transforms if test data
            if split == 'test':
                # testset = CIFAR10(datadir, download=False, train=False, transform=test_transform)
                testset = fetch_dataset(split, datadir, 'CIFAR10')['test']
                dataloader = DataLoader(testset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.pin_cuda)

            dataloaders[split] = dataloader

    return dataloaders


def fetch_subset_dataloader(types, datadir, params, batch_num):
    """
    Fetches dataloader objects for a subset of each type of data

    This might be helpful in case I want test-train a few batches before the full sweep

    Args:
        types: (list) has one or more of "train, val, test" depending on the dataset
        datadir: (str) file path containing the raw dataset
        params: (Params object) hyperparameters controlling data loading behavior

    Returns:
        dataloadr: (dict) contains the DataLoader objects for each type in types
    """

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:

            # return trainset subset dataloader
            if split == 'train':
                # trainset = CIFAR10(datadir, download=False, train=True, transform=train_transform)
                trainset = fetch_dataset(split, datadir, 'CIFAR10')['train']
                subset_indices = range(params.batch_size * batch_num)
                trainset_subset = Subset(trainset, subset_indices)
                dataloader = DataLoader(trainset_subset, batch_size=params.batch_size, shuffle=True,
                                    num_workers=params.num_workers, pin_memory=params.pin_cuda)

            # return testset subset dataloader
            if split == 'test':
                # testset = CIFAR10(datadir, download=False,
                #                    train=False, transform=train_transform)
                testset = fetch_dataset(split, datadir, 'CIFAR10')['test']
                subset_indices = range(params.batch_size * batch_num)
                testset_subset = Subset(testset, subset_indices)
                dataloader = DataLoader(testset_subset, batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers, pin_memory=params.pin_cuda)

            dataloaders[split] = dataloader

    return dataloaders

