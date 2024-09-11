import torch
import ipdb
import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections
from common.model_ktpformer import *


from common.loss import *
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from time import time
from common.utils import *
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.append("/pub/bjvela/KTPFormer/common")  # Add the directory, not the full path to the file
# Paths to the model files
model_1_path = '/pub/bjvela/PoseLab3D/model_243_CPN_best_epoch.bin'
model_2_path = '/pub/bjvela/PoseLab3D/checkpoint/CorrectSavedModel.pt'


def load_state_dict_from_checkpoint_1(path):
    # Load the first checkpoint, which contains additional metadata
    checkpoint = torch.load(path, map_location='cpu')
    
    # Extract the model's state_dict
    model_1_state_dict = checkpoint['model_pos']  # 'model_pos' holds the state_dict
    return model_1_state_dict

def load_state_dict_from_checkpoint_2(path):
    # Load the second checkpoint, which contains the full model
    model_2 = torch.load(path, map_location='cpu')
    
    # Extract the state_dict from the full model
    model_2_state_dict = model_2.state_dict()
    return model_2_state_dict

def add_prefix(state_dict, prefix):
    """Adds the prefix to all keys in the state dict."""
    return {prefix + key: value for key, value in state_dict.items()}

def compare_state_dicts(state_dict_1, state_dict_2):
    for key in state_dict_1:
        if key not in state_dict_2:
            print(f"Key {key} found in model 1 but not in model 2.")
            return False
        if not torch.equal(state_dict_1[key], state_dict_2[key]):
            print(f"Mismatch found in key {key}.")
            print(f"Model 1 tensor: {state_dict_1[key]}")
            print(f"Model 2 tensor: {state_dict_2[key]}")
            return False

    for key in state_dict_2:
        if key not in state_dict_1:
            print(f"Key {key} found in model 2 but not in model 1.")
            return False

    print("The state_dicts of both models are identical.")
    return True


def main():
    # Load state_dicts from both models
    model_1_state_dict = load_state_dict_from_checkpoint_1(model_1_path)
    model_2_state_dict = load_state_dict_from_checkpoint_2(model_2_path)

    # Add "module." prefix to model 2 for comparison
    model_2_state_dict = add_prefix(model_2_state_dict, "module.")

    # Compare the two state_dicts
    compare_state_dicts(model_1_state_dict, model_2_state_dict)

if __name__ == "__main__":
    main()
