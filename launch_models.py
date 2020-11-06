""" Run multiple models """

import argparse
import os

from common import utils
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
    models = [
        #'resnet18',
        #'resnet34',
        #'resnet50',
        #'resnet101',
        #'resnet152',
        #'mobilenet20_1p0_t4',
        #'mobilenetv1_28_1p25_32',
        #'mobilenetv1_28_0p75_32',
        #'shufflenetv1_50_s1p0_g1',
        #'shufflenetv1_50_s1p0_g8',
        #'shufflenetv1_50_s0p5_g1',
        #'shufflenetv1_50_s0p5_g1',
        #'shufflenetv1_50_s0p5_g8',
        #'mnasneta1',
        #'shufflenetv2_51_s0p5',
        #'shufflenetv2_51_s1p5',
        'resnext50_32x4d',
        #'wide_resnet50_2',
        #'mobilenet20_1p0_t3',
        #'mobilenetv1_28_1p0_32',
        #'shufflenetv2_51_s1p0',
        #'shufflenetv1_50_s1p0_g2',
        #'densenet40_k12',
        #'densenet100_k12',
        #'densenet100_k24',
        #'densenetbc100_k12',
        #'densenetbc250_k24',
        #'densenetbc190_k40'
    ]

    for model in models:
        # Modify the relevant parameter in params
        params.model = model

        # Launch job (name has to be unique)
        job_name = "{}".format(model)
        launch.launch_training_job(args.exp_dir, args.data_dir, job_name, params)
