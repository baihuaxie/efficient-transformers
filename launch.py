"""
    create run directories under ./experiments folder
"""

import os
import json
import argparse
from subprocess import check_call
import sys

from common.utils import Params, match_dict_by_value, dict_to_list

PYTHON = sys.executable


# put run directories in a global list for launch script to access
global runs
runs = []

# commandline arguments; normally just use defaults
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='./experiments/', \
    help='Parent directory for all experiments')
parser.add_argument('--data_dir', default='./data/', help="Parent \
    directory for all datasets")
parser.add_argument('--default_dir', default='./common/', \
    help='Directory containing default parameters.json file')
parser.add_argument('--jobs_dir', default='./', \
    help='Directory containing jobs.json file')


# decorator
def register(func):
    """
    Decorator: register a run directory returned by a call to create_job()
    """
    def _wrapper(*args, **kwargs):
        """
        inner decorator function takes arguments
        - no need to modify behaviors so no wrappers
        """
        runs.append(func(*args, **kwargs))
    return _wrapper


@register
def creat_job(run_dct, defaults):
    """
    create a run directory from run_dict

    Args:
        run_dct: (dict) contains parameters for one job run
        defaults: (dict) default keyword arguments for all optimizers, schedulers, datasets, etc.
                  unless specified by run_dct, otherwise always use default keyword arguments
    Return:
        create a run directory if does not exist
        create a self-contained runset.json file under the run directory for main.py
        returns run_dir: os.path object to the directory
    """
    # construct runset.json
    runset = {}
    runset.update({k:v for k, v in defaults.dict.items() if k not in ['data', \
        'optimizer', 'scheduler']})
    for key, value in run_dct.items():
        if key == 'data':
            data_dct = match_dict_by_value(defaults.data, 'dataset', value['dataset'])
            data_dct['kwargs'].update({k:v for k, v in value.items() if k != 'dataset'})
            runset.update({'data': data_dct})
        elif key == 'optimizer':
            optim_dct = match_dict_by_value(defaults.optimizer, 'type', value['type'])
            optim_dct['kwargs'].update({k:v for k, v in value.items() if k != 'type'})
            runset.update({'optimizer': optim_dct})
        elif key == 'lr':
            lr_dct = match_dict_by_value(defaults.scheduler, 'type', value['type'])
            lr_dct['kwargs'].update({k:v for k, v in value.items() if k != 'type'})
            runset.update({'scheduler': lr_dct})
        else:
            runset.update({key:value})
    # create run directory
    run_dir = os.path.join(exp_dir, '_'.join(dict_to_list(run_dct)))
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # save runset.json
    with open(os.path.join(run_dir, 'runset.json'), 'w') as fp:
        json.dump(runset, fp, indent=4)

    # append run directory to global list
    return run_dir


if __name__ == '__main__':
    args = parser.parse_args()
    # read parameters of all job runs from jobs.json
    json_path = os.path.join(args.jobs_dir, 'jobs.json')
    jobs = Params(json_path)

    # read default parameters
    defaults = Params(os.path.join(args.default_dir, 'parameters.json'))

    # create parent experiment directory
    exp_dir = os.path.join(args.exp_dir, jobs.jobname)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # save jobs.json
    jobs.save(os.path.join(exp_dir, 'jobs.json'))

    # create run directories + runset.json files
    for run_dct in jobs.experiments:
        creat_job(run_dct, defaults)

    # Launch job
    for run_dir in runs:
        # launch training job with specified setting
        cmd = "{python} main.py --run_dir={run_dir} --data_dir {data_dir} \
            --run_mode train".format(python=PYTHON, run_dir=run_dir, data_dir=args.data_dir)
        print(cmd)
        check_call(cmd, shell=True)
    