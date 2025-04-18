import argparse
import torch

import sys
import yaml

from dataset import H36M, GPA, Surreal, ThreeDPW
from model import Model
from evaluation import Evaluation


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    #print(type(config))
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Script requires a YAML configuration file.")
    
    parser.add_argument(
        #'--cfg', 
        '--config', 
        required=True, 
        type=str, 
        help='Path to the YAML config file. Example: python readFile.py "--config or -cfg" my_Model_Config.yaml'
    )
    args = parser.parse_args()
    return args


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parse_args()                  # Parse the command-line argument for the config file
    config = load_config(args.config)    # Load the config from the specified YAML file
    
    
    print("Evaluating!")
    #print('python ' + ' '.join(sys.argv))
    print("CUDA Device Count: ", torch.cuda.device_count())
    
    # Sort the config dictionary by keys and print in alphabetical order
    def print_config_namespace_style(config):
        config_items = [f"{key}={repr(value)}" for key, value in sorted(config.items())]
        print("Namespace(" + ", ".join(config_items) + ")")
    print_config_namespace_style(config)

    model = Model(config)

    if config['dataset'] == 'h36m':
        dataset = H36M(config)
    elif config['dataset'] == 'gpa':
        dataset = GPA(config)
    elif config['dataset'] == 'surreal':
        dataset = Surreal(config)
    elif config['dataset'] == '3dpw':
        dataset = ThreeDPW(config)
    # dataset.load_data()


    evaluation = Evaluation(model, dataset)
    evaluation.evaluate()

if __name__ == "__main__":
     main()
