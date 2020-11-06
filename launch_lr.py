"""Peform hyperparemeters search"""

import argparse
import os

import utils
import launch


parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='./experiments/launch-test',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='./data/', help="Directory containing the dataset")


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.exp_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    learning_rates = [1e-4, 1e-3, 1e-2]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch.launch_training_job(args.exp_dir, args.data_dir, job_name, params)
