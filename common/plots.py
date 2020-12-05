"""
utilities for plotting statistics gathered during training
"""

import csv
import os.path as op
import pandas as pd
import seaborn as sns

from common.utils import Params


def save_batch_summary(run_dir, data):
    """
    save batch statistics into .csv file

    Args:
        run_dir: (str) run directory to save csv file; file name = run_dir + 'batch_summary.csv'
        data: (list of dicts) a list of dictionaries; contains key: value pairs for data
    """
    # get headers as all unique keys in data
    headers = set().union(*(d.keys() for d in data))

    # save path
    fpath = op.join(run_dir, 'batch_summary.csv')

    if op.exists(fpath):
        # if csv file exists, append to file (without headers)
        file_mode = 'a'
    else:
        # else create and write to file with headers
        file_mode = 'w'

    with open(fpath, file_mode, newline='') as f:
        dict_writer = csv.DictWriter(f, headers)
        if file_mode == 'w':
            dict_writer.writeheader()
        dict_writer.writerows(data)


def read_csv(csv_file, index_col=None):
    """
    read statistics from csv file into pandas DataFrame objects

    Args:
        csv_file: (str) path to csv file
        index_col: (str) header name to be used as index columnb
    """
    try:
        df = pd.read_csv(csv_file, sep=',', header=0, index_col=index_col)
    except ValueError:
        # index_col does not exist in headers
        df = pd.read_csv(csv_file, sep=',', header=0, index_col=None)
        print("index col '{}' not present in column headers {}".format(index_col, list(df)))
    return df


def read_rc_from_json(rc_path):
    """
    read rc params from json file
    """
    json_path = op.join(rc_path, 'plot_rc.json')
    return Params(json_path)


def plot_batch_summary(save_dir, data, params, xcol=None, metric=None, hue_col=None, \
    prefix=None):
    """
    plot training batch statistics

    Args:
        save_dir: (str) directory to save plots
        data: (pd.DataFrame) pandas DataFrame object which stores statistics for plotting
        params: (Params object) contains plot configurations
        xcol: (str) name of column in 'data' used as x-axis;
              default=None, data must contain a named index col to be used as x-axis
        metrics: (str or list of 2 str's) name(s) of the metric(s), e.g., 'accuracy', 'loss'
        hue_col: (str) column used as hue
        prefix: (str) a string containing information about the data, e.g., train / val, \
            hyper-parameters, etc.

    Note: types of plots depend on data structure
    > need to specify three types of columns:
        - xcol: a single str & not None; used as x-axis
        - metric: a single str or a list of two str's; used as y axis (or axes)
        - hue_col: None or a single str; if speficied, used as hue
    1) xcol + a single metric + no hue
    -  plots a single line plot
    2) xcol + 2 metrics + no hue
    -  this structure implies a plot that uses a 2nd y-axis
    -  e.g., plot accuracy and loss vs. iteration on the same plot
    3) xcol + a single metric + hue
    - e.g., accuracy vs. iteration for different training settings, each colored differently
        by 'hue' column
    4) xcol + 2 metrics + hue
    - same as 3) but with an additional y axis
    """
    if xcol is None:
        assert data.index.name is not None, "x-axis name is not specified by neither xcol \
            nor index column name"
        xcol = data.index.name
        data = data.reset_index()
    # create a line plot
    fig = sns.relplot(data=data, x=xcol, y=metric, kind='line', \
        height=5.0, aspect=1.2)
    # configure axis ranges and labels (use .title() to capitalize first letter)
    fig.set_ylabels(metric.title())
    fig.set_xlabels(xcol.title())
    try:
        fig.set(**params.yaxis[metric])
        fig.set(**params.xaxis[xcol])
    except AttributeError:
        pass
    except KeyError:
        print("x-axis {} or y-axis {} settings not configured".format(xcol, metric))
    # save figure
    fig_title = '_'.join([prefix, metric, 'vs', xcol])
    fig.savefig(op.join(save_dir, fig_title+'.png'))



