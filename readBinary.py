import argparse
import torch
import yaml

from dataset import H36M, GPA, Surreal, ThreeDPW
from model import Model
from evaluation import Evaluation


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Script requires a YAML configuration file.")
    
    parser.add_argument('--config', required=True, type=str, help='Path to the YAML config file. Example: python readFile.py "--config or -cfg" my_Model_Config.yaml')
    parser.add_argument("--dataset", choices=["h36m", "gpa", "surreal", "3dpw"], help="Override the `dataset:` entry in the YAML without editing the file")
    parser.add_argument("--print_sample", action="store_true", help="Print a sample of the input, target, and output data")
    parser.add_argument("--save_predictions", action="store_true", help="Save the predictions to a file")

    args = parser.parse_args()
    return args


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parse_args()                  # Parse the command-line argument for the config file
    config = load_config(args.config)    # Load the config from the specified YAML file

    if args.dataset is not None:
        config["dataset"] = args.dataset
    
    if args.print_sample is True:
        config["print_sample"] = args.print_sample    
    
    if args.save_predictions is True:
        config["save_predictions"] = args.save_predictions
    
    print("==> Evaluating!")
    #print('python ' + ' '.join(sys.argv))
    print("==> CUDA Device Count: ", torch.cuda.device_count())
    
    # Sort the config dictionary by keys and print in alphabetical order
    def print_config_namespace_style(config):
        config_items = [f"{key}={repr(value)}" for key, value in sorted(config.items())]
        print("==> Namespace(" + ", ".join(config_items) + ")")
    print_config_namespace_style(config)

    model = Model(config)
    
    if config['dataset'] == 'h36m':
        dataset = H36M(config['path_to_dataset_h36m'], config)
    elif config['dataset'] == 'gpa':
        dataset = GPA(config['path_to_dataset_gpa'], config)
    elif config['dataset'] == 'surreal':
        dataset = Surreal(config['path_to_dataset_surreal_train'], config['path_to_dataset_surreal_test'], config)
    elif config['dataset'] == '3dpw':
        dataset = ThreeDPW(config['path_to_dataset_3dpw'], config)

    evaluation = Evaluation(model, dataset)
    evaluation.evaluate()

if __name__ == "__main__":
     main()
