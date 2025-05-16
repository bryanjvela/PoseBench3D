
# PoseBench3D

**PoseBench3D** is a unified benchmarking environment for evaluating 2D-to-3D human pose estimation models across multiple datasets (H36M, 3DPW, GPA, SURREAL, etc.). The framework supports PyTorch, ONNX, and TensorFlow-based models, allowing researchers to compare diverse approaches within a single ecosystem.

---

## 📚 Table of Contents
1. [📌 Introduction](#-introduction)  
2. [📦 Environment](#-environment)  
   - [Conda Setup & Dependencies](#conda-setup--dependencies)  
3. [📁 Datasets Overview](#-datasets-overview)  
4. [🧠 Models Overview](#-models-overview)  
5. [📄 Config File (YAML Format)](#-config-file-yaml-format)  
6. [✅ Evaluation](#-evaluation)  
   - [Example Evaluation Commands](#-evaluation)  


---

## 📌 Introduction

PoseBench3D allows you to quickly:

- **Load** multiple datasets (e.g., H36M, 3DPW, GPA, SURREAL).  
- **Evaluate** 2D-to-3D lifting models (including PyTorch JIT-Traced Models and TensorFlow ONNX models).
- **Compare** results under standardized metrics (e.g. MPJPE, PA-MPJPE, per-joint error).  
- **Benchmark** easily with minimal overhead.  

---

## 📦 Environment

We recommend using **conda** + **pip** to ensure an isolated environment that closely matches the tested configuration.

### Conda Setup & Dependencies

1. **Create** a conda environment with Python 3.8:

   ```bash
   conda create -n posebench python=3.8 -y
   conda activate posebench

2. Install PyTorch and torchvision for CUDA 11.1:
   ```bash
   pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

3. Install TensorFlow
   ```bash
   pip install tensorflow==2.12.0

4. Install other required packages from ```requirements.txt```:
   ```bash
   pip install -r requirements.txt

   Note: onnxruntime-gpu==1.17.1 may require CUDA 12.2. If your system only has CUDA 11.1 (or 11.x) available, you might see warnings or fall back to CPU. Ensure you load a module or have drivers that match the required version.
   The above note, regarding onnxruntime-gpu, only applies when planning on running the ONNX-based files that were saved from TensorFlow workflows. 


## 📁 Datasets Overview

Below is a list of the four supported datasets, including their sources and download instructions:

| Code-name | Source                         | Download Instructions |
|-----------|--------------------------------|------------------------|
| **H36M**  | Ionescu *et al.* (2014)        | [Download H36M Test and Train Data](https://drive.google.com/file/d/1c84-AebSywvBhvFF0azF2137tGVYJqOO/view?usp=sharing) → place in in `data/` |
| **GPA**  | Wang *et al.* (2019)    | [Download GPA Test and Train Data](https://drive.google.com/file/d/1KBkD6Uaq0PqSCQTDwfPsMP2C7VG7Pof7/view?usp=sharing) → place in `data/` |
| **3DPW**   | von Marcard *et al.* (2018)             | [Download 3DPW Test and Train Data](https://drive.google.com/file/d/1YNU0wmBsOJF4RtdFsFKljcqEOYDkf79p/view?usp=sharing) → place in `data/` |
| **Surreal** | Varol *et al.* (2017)       | [Download Train](https://drive.google.com/file/d/1sflaTtEmLYBqjKdTcWKp8aXeuWhItXLc/view?usp=sharing), [Download Test](https://drive.google.com/file/d/1sflaTtEmLYBqjKdTcWKp8aXeuWhItXLc/view?usp=sharing) → place both in `data/` |

> ⚠️ Note: If you have the raw data, you can generate the dataset `.npz` files yourself by providing the path to the data using our processing scripts. 


## 🧠 Models Overview

The following models are supported for evaluation within PoseBench3D. Each model was trained on Human3.6M and can be evaluated across 3DPW, GPA, and SURREAL datasets. All models were trained on GT Keypoints. Below are links to the original code repositories and pretrained model files.
### Non-Normalized Models
 **Important**: Please create a directory named `checkpoint/` at the root of the repository and place all downloaded model files (.onnx or .pt) inside it. The evaluation scripts will automatically look there.

| Model Name          | GitHub Repo                                              | Model File (.pt / .onnx)                                  |
|---------------------|-----------------------------------------------------------|------------------------------------------------------------|
| GraFormer           | [GitHub](https://github.com/Graformer/GraFormer)       | [Download Checkpoint](https://drive.google.com/file/d/1wXilD1Xv68IKEMoqzPP5-2MVpr8ZbkG7/view?usp=sharing)     |
| SEM-GCN (Unopt.)    | [GitHub](https://github.com/jfzhang95/PoseAug)         | [Download Checkpoint](https://drive.google.com/file/d/1tp_EP7KQ2C8QAGx3EL14Vg5zZTL_No4d/view?usp=sharing)  |
| SEM-GCN (PoseAug)   | [GitHub](https://github.com/jfzhang95/PoseAug)         | [Download Checkpoint](https://drive.google.com/file/d/1y7K758d-P7jeDlsBEZTo_IfSCGXU1wLx/view?usp=sharing)|
| VideoPose (Unopt.)  | [GitHub](https://github.com/jfzhang95/PoseAug)     | [Download Checkpoint](https://drive.google.com/file/d/18jsYvTaBCtROD7I3O-hqUSFa4u6ZIwTM/view?usp=sharing)|
| GLA-GCN             | [GitHub](https://github.com/bruceyo/GLA-GCN)         | [Download Checkpoint](https://drive.google.com/file/d/18z9hBKV510zIYgZ8bJVRwv1Rpvb-tS21/view?usp=sharing)       |
| ST-GCN (Unopt.)     | [GitHub](https://github.com/jfzhang95/PoseAug)          | [Download Checkpoint](https://drive.google.com/file/d/1zRfVVopzis5daKNIb4GnRCpXf1GQ85Xp/view?usp=sharing)   |
| Martinez (Unopt.)   | [GitHub](https://github.com/jfzhang95/PoseAug)   | [Download Checkpoint](https://drive.google.com/file/d/18Majb5uFeSOGWa-oFCtYOWirZpWY-a5J/view?usp=sharing)      |
| PoseFormer V1       | [GitHub](https://github.com/zczcwh/PoseFormer)   | [Download Checkpoint](https://drive.google.com/file/d/1kkZafVwDAgBDZ9P-JMP0tOkMoCsXEtFx/view?usp=sharing)       |
| PoseFormer V2       | [GitHub](https://github.com/QitaoZhao/PoseFormerV2)   | [Download Checkpoint](https://drive.google.com/file/d/1M6BWwq2rvPHr5mSiFXPnF6sjeHKw-iCa/view?usp=sharing)       |
| KTPFormer           | [GitHub](https://github.com/JihuaPeng/KTPFormer)       | [Download Checkpoint](https://drive.google.com/file/d/1nZ-62jU0LgdN9f19MPFHtZlhJuk5eRFa/view?usp=sharing)           |
| MixSTE              | [GitHub](https://github.com/JinluZhang1126/MixSTE)          | [Download Checkpoint](https://drive.google.com/file/d/1MP_-Mnq27J3xeRlesSwLJM_6kAJuuiJg/view?usp=sharing)              |
| D3DP                | [GitHub](https://github.com/paTRICK-swk/D3DP)            | [Download Checkpoint](https://drive.google.com/file/d/1WDjhLFhFNwI7YJQFT2z_4lEbPzQ7Za1l/view?usp=sharing)                |
| ST-GCN (PoseAug)    | [GitHub](https://github.com/jfzhang95/PoseAug)          | [Download Checkpoint](https://drive.google.com/file/d/1FecyiUOTgKFOy1SiAOXPPbZLIjDWSJKO/view?usp=sharing) |
| MHFormer            | [GitHub](https://github.com/Vegetebird/MHFormer)        | [Download Checkpoint](https://drive.google.com/file/d/1Ck6jgLPrQWqlvqwmrxQyPfH4HreXOv03/view?usp=sharing)            |
| Martinez (PoseAug)  | [GitHub](https://github.com/jfzhang95/PoseAug)   | [Download Checkpoint](https://drive.google.com/file/d/1AEOwm7iu-6VW9hRFkAECH5HrAcMGZGv_/view?usp=sharing)    |
| VideoPose (PoseAug) | [GitHub](https://github.com/jfzhang95/PoseAug)     | [Download Checkpoint](https://drive.google.com/file/d/1K9hQGQd-muobOUpRDS-veaR_Xv1ybjtq/view?usp=sharing)   |


### Normalized Models

The following models have been trained with normalized 2D inputs (e.g., z-score standardization). These are used to evaluate the impact of input normalization on cross-dataset generalization. Each model was trained on one of the supported datasets using only ground truth 2D keypoints.

> 📂 **Reminder**: Place all normalized model checkpoints inside the `checkpoint/` directory.

| Model Name                              | Model File (.pt / .onnx)                                   |
|----------------------------------------|-------------------------------------------------------------|
| Martinez (Trained on H36M, Normalized)      | [Download Checkpoint](https://drive.google.com/file/d/1JWzkf91K0k-XrsTYXYQCY6Q6nt1tqiws/view?usp=sharing)        |
| Martinez (Trained on GPA, Normalized)       | [Download Checkpoint](https://drive.google.com/file/d/1IH-WSaQbHtP75ku-6dIlN6V-B6Z-GzwM/view?usp=sharing)         |
| Martinez (Trained on 3DPW, Normalized)      | [Download Checkpoint](https://drive.google.com/file/d/1cP5rhK4mNASOZn6HI3Tht8DMwAUnDw_a/view?usp=sharing)        |
| Martinez (Trained on SURREAL, Normalized)   | [Download Checkpoint](https://drive.google.com/file/d/16XqK5GMbuM4KVcP82ZjWaMI10uMP__gj/view?usp=sharing)     |
| SEM-GCN (Trained on H36M, Normalized)       | [Download Checkpoint](https://drive.google.com/file/d/1PA-kDD587OsSVm6ktScFGd4Wkgg5fQB8/view?usp=sharing)          |
| SEM-GCN (Trained on GPA, Normalized)        | [Download Checkpoint](https://drive.google.com/file/d/1LWQe-XLREAOqWuC5qzzGq7_HWobT89Zh/view?usp=sharing)           |
| SEM-GCN (Trained on 3DPW, Normalized)       | [Download Checkpoint](https://drive.google.com/file/d/12hfkzq1GrvSWuuLZQFj7ZFW40VgVJy_E/view?usp=sharing)          |
| SEM-GCN (Trained on SURREAL, Normalized)    | [Download Checkpoint](https://drive.google.com/file/d/1tfgaR8Oksy25nE953K13K1vTg7fH54lh/view?usp=sharing)       |





## 📄 Config File (YAML Format)

PoseBench3D uses a unified YAML-based configuration file to manage model settings, dataset paths, runtime behavior, and I/O tensor shapes. This allows users to customize their experiments by simply modifying a single file.

> ⚠️ Note: We provide pre-configured Config YAML files used for our experiments on the models in the ```configs/``` folder.  

---
```yaml
# General settings
dataset:        [h36m, surreal, 3dpw, gpa]
keypoints:      ["gt": ground truth]
checkpoint:     "checkpoint"
evaluate:       "/path_to_checkpoint.pt"
model_type:     [JIT-Traced PyTorch File, ONNX TensorFlow File]

# Dataset locations 
path_to_h36m:            path_to_h36m
path_to_gpa:             path_to_gpa
path_to_surreal_train:   path_to_surreal_train
path_to_surreal_test:    path_to_surreal_test
path_to_3dpw:            path_to_3dpw
data_dir:                path_to_data_directory

# Runtime & miscellaneous options
num_workers:        2-4
print_sample:       [True, False]
per_joint_error:    [True, False]
save_predictions:   [True, False]

# Input tensor shape
input_shape:
  batch_size:       [1...N]
  num_frames:       1
  num_joints:       [14, 16]
  coordinates:      2

# Output tensor shape
output_shape:
  batch_size:       [1...N]
  num_frames:       1
  num_joints:       [14, 16]
  coordinates:      3

# Model-specific flags
model_info:
  flattened_coordinate_model:       [True, False]
  output_3d:                        [Meters, Millimeters]
  root_center_2d_test_input:        [True, False]
  normalize_2d_to_minus_one_to_one: [True, False]
  trained_on_normalized_data:       [True, False]
  video_model:                      [True, False]
  


```
## ✅ Evaluation

To evaluate a given model on a target dataset, use the `readBinary.py` script with the appropriate configuration file. We provide pre-configured Config YAML files used for our experiments on the models in the configs/ folder. Below are two example commands:

**KTPFormer on Human3.6M**  
```bash
python readBinary.py --config configs/Not_Trained_With_Normalization/KTPFormer.yaml --dataset h36m --print_sample
```

**GLA-GCN on GPA**
  ```bash
python readBinary.py --config configs/Not_Trained_With_Normalization/glaConfig.yaml --dataset gpa --print_sample
```

**MixSTE on 3DPW**
  ```bash
python readBinary.py --config configs/Not_Trained_With_Normalization/MixSTE.yaml --dataset 3dpw --print_sample
```
**D3DP on Surreal**
  ```bash
python readBinary.py --config configs/Not_Trained_With_Normalization/D3DP.yaml --dataset 3dpw --print_sample
```

> The --print_sample flag will output one representative prediction to verify the input/output consistency.
> Ensure all dataset paths and model checkpoints are correctly specified in your config file before running evaluation.








   
