import torch
import numpy as np
from common.loss import mpjpe, p_mpjpe, procrustes

from model import Model  # Import Model from model.py
from dataset import Dataset  # If Dataset is from dataset.py
from torch.utils.data import DataLoader
from common.utils import AverageMeter
from common.generators import Evaluate_Generator, PoseGenerator
import time

from progress.bar import Bar
from poseutils.props import calculate_limb_lengths, calculate_avg_limb_lengths
from poseutils.view import draw_axes, draw_skeleton, draw_bounding_box

from matplotlib import pyplot as plt



EDGE_NAMES_16JNTS = ['HipRhip',
              'RFemur', 'RTibia', 'HipLHip',
              'LFemur', 'LTibia', 'LowerSpine',
              'UpperSpine', 'NeckHead',
              'LShoulder', 'LHumerus', 'LRadioUlnar',
              'RShoulder', 'RHumerus', 'RRadioUlnar']


class Evaluation:
    def __init__(self, model: Model, dataset: Dataset):
        self.model = model
        self.dataset = dataset

    def execute_evaluation(self, data_loader, model_pos, device, action=None):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0

        printTargetAndInputShape = True
        printOutputShape = True
        printOneTargetAndOutputDataPoint = True
        
        # Switch to evaluate mode
        with torch.no_grad():
            model_pos = self.model.get_model()
            model_pos.eval()
            N = 0
            
            for targets_3d, inputs_2d in data_loader:    
                # inputs_2d  = inputs_2d.flatten(start_dim=1).to(device)    
                # targets_3d = targets_3d.flatten(start_dim=1)
                if printTargetAndInputShape:
                    print("Inputs 2D Shape: ", inputs_2d.shape)
                    print("Target 3D Shape: ", targets_3d.shape)
                    printTargetAndInputShape = False
                    # exit(0)

                inputs_2d = inputs_2d.to(device)
                outputs_3d = model_pos(inputs_2d).cpu()

                # print(outputs_3d[63].cpu().numpy())
                if self.model.config['zero_center_root'] is True:
                    if self.model.config['video_model'] is True:
                        targets_3d -= targets_3d[:, :, :1, :]
                        outputs_3d -= outputs_3d[:, :, :1, :]
                    else:
                        outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-center the root original non video model 
                # print(outputs_3d[63].cpu().numpy())
                # exit(0)
                if printTargetAndInputShape:
                    print("Target 3D Shape: ", targets_3d.shape)
                    print("Inputs 2D Shape: ", inputs_2d.shape)
                    printTargetAndInputShape = False
                    # exit(0)

                if printOutputShape:
                    print("Outputs 3D Shape: ", outputs_3d.shape)
                    printOutputShape = False
                    #exit(0)

                if printOneTargetAndOutputDataPoint: 
                    inputs_2d_first = inputs_2d[63].cpu().numpy()
                    target_3d_first = targets_3d[63].cpu().numpy()
                    outputs_3d_first = outputs_3d[63].cpu().numpy()
                    print("Input Data Point: \n", inputs_2d_first)
                    print("---------------")
                    print("Target Data Point: \n", target_3d_first)
                    print("---------------")
                    print("Output Data Point: \n", outputs_3d_first)
                    #exit(0)
                    printOneTargetAndOutputDataPoint = False
                    # np.save("target_3d_first_GPA_SEM.npy", target_3d_first)
                    # np.save("output_3d_first_GPA_SEM.npy", outputs_3d_first)

                error = mpjpe(outputs_3d, targets_3d)
                epoch_loss_3d_pos += targets_3d.shape[0] * targets_3d.shape[1] * error.item()
                N += targets_3d.shape[0] * targets_3d.shape[1]

                inputs = targets_3d.cpu().numpy().reshape(-1, targets_3d.shape[-2], targets_3d.shape[-1])
                predicted_3d_pos = outputs_3d.cpu().numpy().reshape(-1, targets_3d.shape[-2], targets_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += targets_3d.shape[0] * targets_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)
        
        if action is None:
            print('----------')
        else:
            print('----' + action + '----')
        e1 = (epoch_loss_3d_pos / N) * 1000
        e2 = (epoch_loss_3d_pos_procrustes / N) * 1000

        #print('Test time augmentation:', test_generator.augment_enabled())
        print(f'Protocol #1 Error (MPJPE): {e1:.2f} mm')
        print(f'Protocol #2 Error (P-MPJPE): {e2:.2f} mm')
        print('----------')
        return e1, e2
    

    def evaluate(self):
        print('\nEvaluating...')
        
        all_actions = None
        if self.model.config['dataset'] == 'h36m':
            all_actions, _ = self.dataset.organize_actions()
            poses_3d_all, poses_2d_all = self.dataset.data(all_actions)
        
        if self.model.config['dataset'] in {'gpa', 'surreal', '3dpw'}:
            poses_3d_all, poses_2d_all = self.dataset.data()
            offset_range = slice(None)
            poses_3d_all[:, offset_range] -= poses_3d_all[:, :1]  # Zero-center the root

        
        
        print(f"3D Target Data Shape: {poses_3d_all.shape}") 
        print(f"2D Input Data Shape: {poses_2d_all.shape}\n") 

        # # Extract a single frame (first sample)
        # single_pose_3d = poses_3d_all[0]  # Shape: (16, 3) or (14, 3)
        # single_pose_2d = poses_2d_all[0]  # Shape: (16, 2) or (14, 2)

        # print(f"Single 3D Pose Datapoint: {single_pose_3d}")

        # print(f"Single 3D Pose Shape: {single_pose_3d.shape}")
        # # print(f"Single 2D Pose Shape: {single_pose_2d.shape}")

        # # Convert to numpy arrays (in case they are tensors)
        # single_pose_3d = np.array(single_pose_3d)
        # # single_pose_2d = np.array(single_pose_2d)

        # # Compute limb lengths
        # limb_lengths_3d = calculate_limb_lengths(single_pose_3d)  # 3D limb lengths
        # avg_limb_lengths_3d, _, _ = calculate_avg_limb_lengths(poses_3d_all)
        # limb_lengths_3d = np.array(avg_limb_lengths_3d) * 1000  # Now this should work

        # # limb_lengths_2d = np.array(limb_lengths_2d) * 1000

        # print("3D Limb Lengths:", limb_lengths_3d)

        # print("\n===== 3D Limb Lengths =====")
        # for name, length in zip(EDGE_NAMES_16JNTS, limb_lengths_3d):
        #     print(f"{name}: {length:.2f} mm")
        # # print("2D Limb Lengths:", limb_lengths_2d)
        # exit(0)
        
        # Prepare data for validation
        valid_loader = DataLoader(
                                  PoseGenerator(poses_3d_all, poses_2d_all,video_mode=self.model.config['video_model']),
                                  batch_size=self.model.config['input_shape'].get('batch_size'), 
                                  shuffle=False,
                                  num_workers=self.model.config['num_workers'], 
                                  pin_memory=True)

        self.execute_evaluation(valid_loader, self.model, torch.device("cuda"), action=None)
        exit(0)

        

