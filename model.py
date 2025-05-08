import os
from typing import Dict
import torch
import torch.nn as nn
import numpy as np

# For ONNX
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.log_severity_level = 4  

class Model:
    def __init__(self, config: Dict):
        if not isinstance(config, dict):
            raise TypeError(f"Expected 'config' to be of type 'dict', but got {type(config).__name__}")

        self.config = config
        self.model_info = config['model_info']
        self.model_type = config.get('model_type', 'Pytorch')  # "JIT", "Pytorch", or "ONNX"

        # If you have extra setup logic for frames/padding, you can keep it here
        if config.get('number_of_frames', None):
            self.receptive_field = config['number_of_frames']
            print("==> Receptive field: ", self.receptive_field)
        
        # Path to the model file (either .pt / .pth / .onnx / .jit, etc.)
        model_path = os.path.join(config['checkpoint'], config['evaluate'])
        print('==> Loading model from', model_path)

        if self.model_type == 'ONNX':
            ###########################################
            # 1) Load ONNX model with ONNX Runtime
            ###########################################
            print("==> Active providers:", ort.get_available_providers())
            print("==> Loading ONNX model")
            self.session = ort.InferenceSession(model_path, sess_options, providers=['CUDAExecutionProvider'])
            self.ort_input_name = self.session.get_inputs()[0].name
            print("==> Loaded ONNX model")
            model_params = None

        elif self.model_type == 'JIT':
            ###########################################
            # 2) Load a TorchScript (JIT) model
            ###########################################
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = torch.jit.load(model_path, map_location=device)

            self.model.eval()

            # Count parameters
            model_params = sum(p.numel() for p in self.model.parameters())

            if torch.cuda.is_available():
                self.model = self.model.cuda()

        else:
            ###########################################
            # 3) Load a standard PyTorch checkpoint
            ###########################################
            # Example: 'model.pth' or 'model.pt'
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = torch.jit.load(model_path, map_location=device)

            self.model.eval()

            # Count parameters
            model_params = sum(p.numel() for p in self.model.parameters())

            if torch.cuda.is_available():
                self.model = self.model.cuda()

        # Final print statement
        if model_params is None:
            print("==> INFO: Trainable parameter count: Either Model is ONNX or Not Available.")
        else:
            print('==> INFO: Trainable parameter count:', model_params / 1e6, 'Million')

    def predict(self, inputs):
        """
        Perform a forward pass. 
        For ONNX, inputs should be a NumPy array. 
        For PyTorch, inputs can be a torch.Tensor (or a NumPy array that we convert).
        """
        if self.model_type == 'ONNX':
            # 1) We do inference using onnxruntime session
            # Ensure input is a NumPy array (float32)
            if not isinstance(inputs, np.ndarray):
                inputs = inputs.numpy()  # or handle as needed
            inputs = inputs.astype(np.float32)

            outputs = self.session.run(None, {self.ort_input_name: inputs})
            # outputs is a list of all output arrays from the ONNX graph
            # Usually there's just 1 output
            return outputs[0]

        else:
            # 2) PyTorch (JIT or normal)
            # Make sure inputs are Tensors
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs)
            inputs = inputs.float()  # ensure float32

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            with torch.no_grad():
                preds = self.model(inputs)
            return preds

    def get_model(self):
        """
        Return the underlying model object.
        - If ONNX, that's your onnxruntime session.
        - If PyTorch, that's the nn.Module or ScriptModule.
        """
        if self.model_type == 'ONNX':
            return self.session
        else:
            return self.model
