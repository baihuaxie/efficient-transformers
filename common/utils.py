"""
    Contains utility classes / functions

    - class Params(): loading / saving parameters from / to a json file
    - set_logger: set the logger to log info in terminal and into 'log_path'

"""

import json
import logging
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
from torchsummary import summary

class Params():
    """
    Class definition for loading hyperparameters from a json file
    - json file needs to contain dict-like definitions for hyperparameters, e.g., {"learning_rate": 0.001}

    Example:

    params = Params(/path/to/json)
    print(params.learning_rate)

    """

    def __init__(self, json_path=None):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """ save parameters to json file """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        
    def update(self, json_path):
        """ update parameters from json file """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    @property
    def dict(self):
        """ return hyperparameters as a dictionary by Params.dict['hyperparameter name'] """
        return self.__dict__


class RunningAverage():
    """
    A class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg = loss_avg.update(5)
    loss_avg = loss_avg.upate(7)
    print(loss_avg()) -> 6 

    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        """ updates total by adding val to total and increase steps by 1 """
        self.steps += 1
        self.total += val

    def __call__(self):
        return float(self.total) / float(self.steps)


def set_logger(log_path):
    """
    Set the logger to log info in terminal and into 'log_path'

    Save every output to the terminal into a permant file

    Args:
        log_path: (string) path to log file
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # log into a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # log into a console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)



def load_checkpoint(checkpoint: str, model, optimizer=None):
    """
    Load model parameters (state_dict) from file. If optimizer is provided, load state_dict for optimizer.

    Args:
        checkpoint: (string) filename for loading parameters
        model: (torch.nn.Module) model object for which the parameters are loaded into
        optimizer: (torch.optim) optional - optimizer object to resume from checkpoint

    Return:
        checkpoint: (dict) a dictionary object containing state_dict, optim_dict produced by torch.load(file)

    """
    if not os.path.exists(checkpoint):
        raise("Checkpoint file doesn't exist: {}".format(checkpoint))
    
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_checkpoint(state, is_best, checkpoint):
    """
    Saves model and training states as state_dict into checkpoint + 'last.pth.zip'
    If is_best == True, also saves into checkpoint + 'best.pth.zip'

    Args:
        state: (dict) contains model's state_dict or other states (e.g., optimizer, epoch)
        is_best: (Boolean) True if the states represent the best model (by some metrics) so far
        checkpoint: (str) folder name used to save states file(s)

    """

    filepath = os.path.join(checkpoint, 'last.pth.zip')
    if not os.path.exists(checkpoint):
        print("Checkpoint directory does not exist, making directory: {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.zip'))


def matplotlib_imshow(img, one_channel=False):
    """ 
    helper function to show an image

    Args:
        img: (tensor) 2D image

    """
    if one_channel:
        img = img.mean(dim=0)
    # un-normalize
    img = img / 2 + 0.5 
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def print_net_summary(log, net, input):
    """ 
    Print the net summary into a log file 

    Args:
        log: (str) log file path
        net: (nn.Module) network instance
        input: (torch.Tensor) input tensor of size [batch, channel, ...]

    """

    original_stdout = sys.stdout
    with open(log, 'w') as f:
        sys.stdout = f
        summary(net, input_size=tuple(input.size()[1:]))
    sys.stdout = original_stdout
    f.close()


def match_dict_by_value(lst, key, value):
    """
    match a dict by a specific key-value pair

    Args:
        lst: (list of dicts) a list of dicts, all contains a common key but may have different values
        key: (str) common key
        value: the value used to find the spcific dict

    Returns:
        dct: (dict) a dictionary element of lst that has the matched key:value pair
    """
    for dct in lst:
        assert isinstance(dct, dict)
        if key in dct.keys():
            if dct[key] == value:
                return dct
    raise ValueError("key-value pair {}:{} not found in settings {}".format(key, value, lst))


def dict_to_list(dct):
    """
    put all values in a dictionary (could by hierarchical) into a list object
    also convert all elements into str
    """
    assert isinstance(dct, dict)
    lst = []
    for _, value in dct.items():
        if isinstance(value, dict):
            lst += [str(v) for v in value.values()]
        else:
            lst.append(str(value))
    return lst
