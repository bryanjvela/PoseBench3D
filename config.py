import numpy as np

from common.graph_utils import adj_mx_from_skeleton, adj_mx_from_skeleton_temporal  # Import the functions


# Manually input the full paths
MODEL_CONFIG = {
    "model_class": "KTPFormer",
    "model_file": "/pub/bjvela/PoseLab3D/model_243_CPN_best_epoch.bin",
    "module": "model_ktpformer",
    "model_args": {
        "adj": None,
        "adj_temporal": None,
        "num_frame": 243,
        "num_joints": 17,
        "in_chans": 2,
        "embed_dim_ratio": 512,
        "depth": 7,
        "num_heads": 8,
        "mlp_ratio": 2.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_path_rate": 0.0
    }
}

# Function to adjust based on GPU availability
def adjust_for_gpu():
    import torch
    if torch.cuda.is_available():
        MODEL_CONFIG["device"] = "cuda"
    else:
        MODEL_CONFIG["device"] = "cpu"

# Function to construct adjacency matrices
def construct_adjacency_matrices(dataset, num_frames):
    print("Constructing adjacency matrices...")
    
    # Temporal skeleton construction
    temporal_skeleton = list(range(0, num_frames))
    temporal_skeleton = np.array(temporal_skeleton)
    temporal_skeleton -= 1

    # Construct adj_temporal
    adj_temporal = adj_mx_from_skeleton_temporal(num_frames, temporal_skeleton)
    
    # Construct adj using dataset skeleton method
    adj = adj_mx_from_skeleton(dataset.skeleton())

    # Update model arguments with adjacency matrices
    MODEL_CONFIG["model_args"]["adj"] = adj
    MODEL_CONFIG["model_args"]["adj_temporal"] = adj_temporal

    print("Adjacency matrices constructed.")

# Call this function to adjust based on the environment
adjust_for_gpu()

# Main block to allow for testing/debugging
if __name__ == "__main__":
    print("Model Configuration:", MODEL_CONFIG)
