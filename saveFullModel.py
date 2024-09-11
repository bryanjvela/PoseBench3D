import sys
import os
import torch

# # Add the relevant directory to Python's path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("/pub/bjvela/KTPFormer/common")  # For testing purposes

from config import MODEL_CONFIG, construct_adjacency_matrices

# Dataset loading configuration
def load_dataset(args):
    print('Loading dataset...')
    dataset_path = 'data/data_3d_' + args['dataset'] + '.npz'

    if args['dataset'] == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    elif args['dataset'].startswith('humaneva'):
        from common.humaneva_dataset import HumanEvaDataset
        dataset = HumanEvaDataset(dataset_path)
    elif args['dataset'].startswith('custom'):
        from common.custom_dataset import CustomDataset
        dataset = CustomDataset('data/data_2d_' + args['dataset'] + '_' + args['keypoints'] + '.npz')
    else:
        raise KeyError('Invalid dataset')

    return dataset

def remove_module_prefix(state_dict):
    """Removes the 'module.' prefix from state_dict keys."""
    new_state_dict = {}
    for key in state_dict:
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict

def load_model():
    # Dynamically import the model module and class
    module_name = MODEL_CONFIG['module']
    module = __import__(module_name, fromlist=[MODEL_CONFIG['model_class']])
    model_class = getattr(module, MODEL_CONFIG['model_class'])

    # Extract model arguments, including adj and adj_temporal
    model_args = MODEL_CONFIG.get("model_args", {})

    # Instantiate the model with the required arguments
    model = model_class(**model_args)
    
    # Load the state dictionary
    state_dict = torch.load(MODEL_CONFIG['model_file'], map_location=torch.device(MODEL_CONFIG.get('device', 'cpu')))
    
    # Check if the state_dict contains the 'model_pos' key and extract it
    if 'model_pos' in state_dict:
        state_dict = state_dict['model_pos']
    
    # Remove the 'module.' prefix if present
    state_dict = remove_module_prefix(state_dict)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=True)
    
    # Save the entire model (optional)
    torch.save(model, 'CorrectSavedModel.pt')
    
    model.eval()
    return model

def compare_state_dicts(state_dict1, state_dict2):
    """Compares two state dictionaries and prints if they match or not."""
    for key in state_dict1.keys():
        if key not in state_dict2:
            print(f"Key {key} found in state_dict1 but not in state_dict2.")
            return False
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"Mismatch found in key {key}.")
            return False
    print("State dictionaries match!")
    return True

def get_user_dataset_choice():
    # Prompt the user to input which dataset they'd like to test on
    print("Please choose a dataset for testing:")
    print("1. Human3.6M")
    print("2. 3DPW")
    print("3. MPI-INF-3DHP")
    print("4. Custom Dataset")

    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        return "h36m"
    elif choice == '2':
        return "3DPW"
    elif choice == '3':
        return "MPI-INF-3DHP"
    elif choice == '4':
        return input("Enter the name of your custom dataset: ")
    else:
        print("Invalid choice, defaulting to Human3.6M.")
        return "h36m"

def main():
    # Get the user's dataset choice and update the dataset configuration
    dataset_name = get_user_dataset_choice()
    print(f"Dataset selected: {dataset_name}")

    # Prepare the dataset arguments
    dataset_args = {
        "dataset": dataset_name,
        "keypoints": "cpn_ft_h36m_dbb"  # Modify this value as needed
    }

    # Load the dataset
    dataset = load_dataset(dataset_args)
    print(f"Dataset loaded: {dataset}")
    
    # Construct adjacency matrices based on the loaded dataset and the number of frames
    construct_adjacency_matrices(dataset, MODEL_CONFIG["model_args"]["num_frame"])

    # Load the model
    model = load_model()
    print("Model loaded successfully!")

    # Load the saved model and compare state_dicts
    saved_model = torch.load('CorrectSavedModel.pt')
    match = compare_state_dicts(model.state_dict(), saved_model.state_dict())
    
    if match:
        print("Model state_dict matches the saved model.")
    else:
        print("Model state_dict does not match the saved model.")

if __name__ == "__main__":
    main()
