""" script to launch train.py with commandline arguments """


from subprocess import check_call
import sys
import os

PYTHON = sys.executable
def launch_training_job(train_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        train_dir: (string) directory for training experiments (containing params, weights and logs)
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    exp_dir = os.path.join(train_dir, job_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Write parameters in json file
    json_path = os.path.join(exp_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --exp_dir={exp_dir} --data_dir {data_dir} --model {model} --run_mode train".format(python=PYTHON, \
                exp_dir=exp_dir, data_dir=data_dir, model=params.model)
    print(cmd)
    check_call(cmd, shell=True)
