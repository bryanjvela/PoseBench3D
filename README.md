# PoseLab3D

A general testing environment for 3D Pose Estimation.

## Table of Contents

- [Introduction](#introduction)
- [Environment](#environment)
- [Usage](#usage)
- [Datasets & Models](#datasets--models)
- [Evaluating](#evaluating)

## Introduction

This project provides a general testing environment for researchers to evaluate their 3D pose estimation models on a variety of compatible datasets, all within a unified framework. The goal is to create a one-stop shop for testing various models without the need for repetitive setup.

## Environment

The code is developed and tested in the following environment:

- Python 3.8
- PyTorch 1.8
- CUDA 11.4.0

To set up a similar environment, follow the instructions below.

## Usage

### Installation

1. Clone this repository
2. Set up the conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate PoseLab
   ```

## Datasets & Models

At present, this framework supports the KTPFormer model. Our goal is to generalize the environment to support other models, including GLA-GCN, and eventually all other models. Each model requires its own `.yaml` configuration file, specifying the model parameters. There’s no need to create a new `.yaml` file unless you intend to test a different model. For now, it’s best to ensure the program works with the existing models without hardcoding.

### Dataset

Create a `data/` directory and place the necessary datasets inside. 

* Human3.6M: [CPN 2D](https://drive.google.com/file/d/1WkbQ1rF1jzeoiwis6YGncLs0f2nprF5A/view?usp=sharing), [Mask R-CNN 2D](https://drive.google.com/file/d/1poB_jys-mOk8OFWg6HUXoy00fk-E-f6t/view?usp=sharing), [Ground-truth 2D](https://drive.google.com/file/d/1btsDnC0RTd_6J1sCtFW5p78u2sbRMcfJ/view?usp=sharing), [Ground-truth 3D](https://drive.google.com/file/d/1F5JG3wvbGd762672G7hXM-tKQ91dQO1c/view?usp=sharing).

Other datasets are still a work in progress and will be added in future updates.

### Models

Create a `checkpoint/` directory and place the following models inside:

- **KTPFormer**: You can download the KTPFormer model from [this link](https://drive.google.com/file/d/1cW-4MxKGM6NE4kLCFWNt2GhHCWAyY_sJ/view?usp=sharing). For more information on KTPFormer, visit the [KTPFormer GitHub repository](https://github.com/JihuaPeng/KTPFormer).
- **GLA-GCN**: You can download the GLA-GCN model from [this link](https://drive.google.com/file/d/1eGObTduC4CeT1OMSoAMYNJ4Rf2a_F9YH/view?usp=sharing). For more information on GLA-GCN, check out the [GLA-GCN GitHub repository](https://github.com/bruceyo/GLA-GCN).

## Evaluating

To evaluate the models, run the following command:

```bash
python readFile.py --config ktpConfig.yaml
```

Ensure that the appropriate configuration file (`ktpConfig.yaml` for KTPFormer or a similar file for other models) is specified.

