import numpy as np
import argparse
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import yaml

from common.camera import *
from common.loss import *
from common.generators import UnchunkedGenerator_Seq
from common.utils import *


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Script requires a YAML configuration file.")
    
    # Provide a more detailed help message with an example usage
    parser.add_argument(
        '-cfg', 
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

    sys.path.append(config['model_location'])  # Add the directory, not the full path to the file

    print("Evaluating!")
    print('python ' + ' '.join(sys.argv))
    print("CUDA Device Count: ", torch.cuda.device_count())
    
    # Sort the config dictionary by keys and print in alphabetical order
    def print_config_namespace_style(config):
        config_items = [f"{key}={repr(value)}" for key, value in sorted(config.items())]
        print("Namespace(" + ", ".join(config_items) + ")")
    print_config_namespace_style(config)

    # Dataset loading 
    print('Loading dataset...')
    dataset_path = 'data/data_3d_' + config['dataset'] + '.npz'
    if config['dataset'] == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
        dataset.preprocess(config)
    else:
        raise KeyError('Invalid dataset')

    # Model API Logic 
    from model import Model
    model_instance = Model(config) # Instantiate the model class  
    model_instance.load_model(config) # Load model with checkpoint
    model_instance.eval_h36m(dataset) # Evaluate model on h36m

if __name__ == "__main__":
     main()
