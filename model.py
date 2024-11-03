import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import os

from typing import Dict
from torch.nn import Module

class Model:
    def __init__(self, config: Dict):
        if not isinstance(config, dict):
            raise TypeError(f"Expected 'config' to be of type 'dict', but got {type(config).__name__}")
        self.config = config
        self.receptive_field = config['number_of_frames']
        self.pad = (self.receptive_field - 1) // 2
        self.causal_shift = 0
        
        model_path = os.path.join(config['checkpoint'], config['evaluate'])
        print('Loading model from', model_path)
        self.model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.eval()
        #print(type(self.model))
        #exit()
        model_params = sum(p.numel() for p in self.model.parameters())
        print('INFO: Trainable parameter count:', model_params/1000000, 'Million')

        # make model parallel (not working for multiple gpus)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
    
    def get_model(self) -> Module:
        return self.model

