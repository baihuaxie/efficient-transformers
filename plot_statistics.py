"""
script to plot standard statistics after training
"""

import os
import json
import argparse

# commandline arguments; normally just use defaults
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='./experiments/', \
    help='Parent directory for all experiments')
parser.add_argument('--jobs_dir', default='./', \
    help='Directory containing jobs.json file')

if __name__ == '__main__':
    args = parser.parse_args()
    # read parameters of all job runs from jobs.json
    json_path = os.path.join(args.jobs_dir, 'jobs.json')
    jobs = Params(json_path)