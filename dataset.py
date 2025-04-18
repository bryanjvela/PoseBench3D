from typing import List, Tuple


from common.skeleton import Skeleton
from common.h36m_dataset import Human36mDataset

import scipy.io
from smpl_relations import get_intrinsic, get_extrinsic

import numpy as np


import argparse
import os
import zipfile
import numpy as np
import h5py
from glob import glob
from shutil import rmtree
import cdflib
import sys
sys.path.append('../')
from common.h36m_dataset import Human36mDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from common.utils import wrap

subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

class Dataset:
    def data(self):# -> Tuple[List[torch.float32], List[torch.float32]]:
        pass
    
    # TODO get kps_left & kps_right & joints left and right 


class H36M(Dataset):
    def __init__(self, config):
        self.config = config
        
        data_3d_dir = os.path.abspath(self.config.get('data_dir', '.'))
        os.makedirs(data_3d_dir, exist_ok=True)
        output_filename_3d = os.path.join(data_3d_dir, 'data_3d_h36m.npz')

        data_2d_dir = os.path.abspath(self.config.get('data_dir', '.'))
        os.makedirs(data_2d_dir, exist_ok=True)
        output_filename_2d = os.path.join(data_2d_dir, 'data_2d_h36m_gt.npz')

        # ------------------------------------------------------------------
        # Prepare the shape metadata for this run (from config).
        # For the 3D dataset, we typically compare against `output_shape`.
        # For the 2D dataset, we typically compare against `input_shape`.
        # ------------------------------------------------------------------
        self.expected_3d_meta = {
            'num_frames':   config['output_shape'].get('num_frames'),
            'num_joints':   config['output_shape'].get('num_joints'),
            'coordinates':  config['output_shape'].get('coordinates'),
        }
        self.expected_2d_meta = {
            'num_frames':   config['input_shape'].get('num_frames'),
            'num_joints':   config['input_shape'].get('num_joints'),
            'coordinates':  config['input_shape'].get('coordinates'),
        }

        # ------------------------------------------------------
        # 1) Check if 3D file exists AND if shape metadata matches
        # ------------------------------------------------------
        create_3d_data = True
        if os.path.isfile(output_filename_3d):
            print(f"3D dataset file {output_filename_3d} already exists.")
            with np.load(output_filename_3d, allow_pickle=True) as data:
                if 'metadata_3d' in data:
                    stored_3d_meta = data['metadata_3d'].item()
                    if self._shapes_match(stored_3d_meta, self.expected_3d_meta):
                        print("3D metadata matches config. Using existing file.")
                        output = data['positions_3d'].item()  # Convert np.object array -> dict
                        create_3d_data = False
                    else:
                        print("3D metadata does NOT match config. Overwriting...")
                else:
                    print("No 3D metadata found. Overwriting...")
        else:
            print(f"3D dataset file {output_filename_3d} does not exist. Creating...")

        # ------------------------------------------------------
        # If needed, create and save the 3D data
        # ------------------------------------------------------
        if create_3d_data:
            print('Creating 3D dataset from Human3.6M (CDF) files...')
            output = {}
            subjects = ['S1','S5','S6','S7','S8','S9','S11']
            for subject in subjects:
                output[subject] = {}
                file_list = glob(os.path.join(self.config['path_to_dataset'], subject, 'MyPoseFeatures', 'D3_Positions', '*.cdf'))
                assert len(file_list) == 30, (f"Expected 30 files for subject {subject}, got {len(file_list)}")
                for f in file_list:
                    action = os.path.splitext(os.path.basename(f))[0]
                    if subject == 'S11' and action == 'Directions':
                        continue  # Discard corrupted video

                    # Use consistent naming convention
                    canonical_name = (action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog'))
                    hf = cdflib.CDF(f)
                    positions = hf['Pose'].reshape(-1, 32, 3)
                    positions /= 1000.0  # Convert mm -> meters
                    output[subject][canonical_name] = positions.astype('float32')

            print('Saving 3D poses...')
            np.savez_compressed(
                output_filename_3d, 
                positions_3d=output,
                metadata_3d=self.expected_3d_meta  # Store the shape metadata in the file
            )
            print(f'3D Dataset saved to {output_filename_3d}')

        # ------------------------------------------------------
        # 2) Build the internal dataset for 3D
        #    (Human36mDataset will read from the saved npz.)
        # ------------------------------------------------------
        print("Initializing internal Human36mDataset for 3D data...")
        self._dataset = Human36mDataset(output_filename_3d, self.config)

        # ------------------------------------------------------
        # 3) Check if 2D file exists AND if shape metadata matches
        # ------------------------------------------------------
        create_2d_data = True
        if os.path.isfile(output_filename_2d):
            print(f"2D dataset file {output_filename_2d} already exists.")
            with np.load(output_filename_2d, allow_pickle=True) as data:
                if 'metadata_2d' in data:
                    stored_2d_meta = data['metadata_2d'].item()
                    if self._shapes_match(stored_2d_meta, self.expected_2d_meta):
                        print("2D metadata matches config. Using existing file.")
                        output_2d_poses = data['positions_2d'].item()
                        metadata = data['metadata'].item() if isinstance(data['metadata'], np.ndarray) else data['metadata']
                        create_2d_data = False
                    else:
                        print("2D metadata does NOT match config. Overwriting...")
                else:
                    print("No 2D metadata found. Overwriting...")
        else:
            print(f"2D dataset file {output_filename_2d} does not exist. Creating...")

        # ------------------------------------------------------
        # If needed, create and save the 2D data
        # ------------------------------------------------------
        if create_2d_data:
            print('\nComputing ground-truth 2D poses...')
            output_2d_poses = {}
            for subject in self._dataset.subjects():
                output_2d_poses[subject] = {}
                for action in self._dataset[subject].keys():
                    anim = self._dataset[subject][action]
                    positions_2d = []
                    for cam in anim['cameras']:
                        pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                        pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                        pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                        positions_2d.append(pos_2d_pixel_space.astype('float32'))
                    output_2d_poses[subject][action] = positions_2d

            print('Saving 2D poses...')
            # You may already have metadata in `metadata`, so let's add
            # your shape fields as a separate 'metadata_2d' dict.
            metadata = {'num_joints': self._dataset.skeleton().num_joints(),
                'keypoints_symmetry': [
                    self._dataset.skeleton().joints_left(),
                    self._dataset.skeleton().joints_right()
                ]
            }
            np.savez_compressed(
                output_filename_2d, 
                positions_2d=output_2d_poses,
                metadata=metadata,         # your existing metadata
                metadata_2d=self.expected_2d_meta  # store shape-based metadata
            )
            print(f'2D Dataset saved to {output_filename_2d}')

        # ------------------------------------------------------
        # 4) Preprocess the dataset with the newly created/loaded 2D data
        # ------------------------------------------------------
        print("Preprocessing dataset with 2D data...")
        self._dataset.preprocess(self.config, output_filename_2d)

        self.kps_left = self._dataset.kps_left
        self.kps_right = self._dataset.kps_right
        self.joints_left = self._dataset.joints_left
        self.joints_right = self._dataset.joints_right
        self.keypoints = self._dataset.keypoints
        print("H36M initialization complete.")


    def organize_actions(self):
        return self._dataset.organize_actions()
    
    def data(self, actions):
        # Flatten the dictionary {action_name: [(subject, action), ...]} 
        # into a single list of (subject, action) pairs
        flat_actions = []
        for _, sa_list in actions.items():
            flat_actions.extend(sa_list)
        
        return self._dataset.fetch_actions(flat_actions, self.config)

    def _shapes_match(self, stored_meta, expected_meta):
        """Compare num_frames, num_joints, and coordinates in metadata."""
        return (
            stored_meta.get('num_frames', None) == expected_meta.get('num_frames', None)
            and stored_meta.get('num_joints', None) == expected_meta.get('num_joints', None)
            and stored_meta.get('coordinates', None) == expected_meta.get('coordinates', None)
        )

class GPA(Dataset):
    def __init__(self, config):
        self.config = config

        print("Loading GPA dataset...")
        data = np.load("/pub/bjvela/PoseLab3D/conformed_GPA_meters.npz", allow_pickle=True)

        # Extract the stored dictionary
        data_obj = data['data'].item()

        # Load test data only
        test_data = data_obj.get('test', {})

        # Extract 2D, 2D normalized, 3D, and 3D normalized test data
        self.test_2d = test_data.get('2d', None)
        self.test_2d_normalized = test_data.get('normalized_2d', None)
        self.test_3d = test_data.get('3d', None)
        self.test_3d_normalized = test_data.get('normalized_3d', None)

        # Print shape information for debugging
        # print(f"Test 2D shape: {self.test_2d.shape if self.test_2d is not None else 'None'}")
        # print(f"Test 2D Normalized shape: {self.test_2d_normalized.shape if self.test_2d_normalized is not None else 'None'}")
        # print(f"Test 3D shape: {self.test_3d.shape if self.test_3d is not None else 'None'}")
        # print(f"Test 3D Normalized shape: {self.test_3d_normalized.shape if self.test_3d_normalized is not None else 'None'}")
        print("Sucessfully loaded GPA dataset.")

    def data(self, actions):
        return self.test_3d, self.test_2d


class Surreal(Dataset):
    def __init__(self, config):
        self.config = config
        self.npz_file = "surreal_preprocessed.npz"

        if os.path.exists(self.npz_file):
            print(f"Loading preprocessed data from {self.npz_file}...")
            data = np.load(self.npz_file)
            self._3d_cam = data['coords_3d']
            self._2d = data['coords_2d']
            self._3d_cam_norm = data['coords_3d_norm']
            self._2d_norm = data['coords_2d_norm']
        else:
            print("Preprocessing SURREAL data from .mat files...")
            base_dir = '../SURREAL/data/cmu/test/run0'
            # Gather all *_info.mat files
            mat_files = []
            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    if f.endswith('_info.mat'):
                        mat_files.append(os.path.join(root, f))

            # Use lists, then one-time concat
            list_3d = []
            list_2d = []

            for mat_path in mat_files:
                data = scipy.io.loadmat(mat_path)
                joints3D = data['joints3D'].transpose(2, 1, 0)  # (N,24,3)
                joints2D = data['joints2D'].transpose(2, 1, 0)  # (N,24,2)
                list_3d.append(joints3D)
                list_2d.append(joints2D)

            self._3d_world = np.concatenate(list_3d, axis=0)
            self._2d       = np.concatenate(list_2d, axis=0)
            
            # Subselect joints
            to_select = [0, 1, 4, 7, 2, 5, 8, 6, 12, 15, 17, 19, 23, 16, 18, 22]
            self._3d_world = self._3d_world[:, to_select, :]
            self._2d       = self._2d[:,       to_select, :]

            # Convert from world to camera
            # (Be sure that `data` still points to the last read,
            #  or re-derive extrinsics in a consistent way.)
            _, R, T = get_extrinsic(data['camLoc'])
            self._3d_cam = np.copy(self._3d_world)
            for i in range(self._3d_cam.shape[0]):
                self._3d_cam[i] = np.dot(R, self._3d_cam[i].T).T + T.T

            # Optional: zero-center or “normalize” in a single pass
            self._3d_cam_norm = self._3d_cam - self._3d_cam[:, :1, :]
            self._2d_norm     = self._2d     - self._2d[:,     :1, :]
            np.savez_compressed("surreal_preprocessed.npz", 
                        coords_3d=self._3d_cam, 
                        coords_2d=self._2d,
                        coords_3d_norm=self._3d_cam_norm,
                        coords_2d_norm=self._2d_norm)
            # TODO
            # normalize_zscore(self.normalized_3d_train, self.mean_3d, self.std_3d, skip_root=True)
            # normalize_zscore(self.normalized_3d_train, self.mean_3d, self.std_3d, skip_root=True)

    def data(self):
        return self._3d_cam, self._2d
        # return self._3d_cam_norm, self._2d


class ThreeDPW(Dataset):
    def __init__(self, config):
        self.config = config

        print("Loading 3DPW dataset...")
        data = np.load("/pub/bjvela/PoseLab3D/test1_3dpw.npz", allow_pickle=True)

        # Extract the stored dictionary
        data_obj = data['data'].item()

        # Load test data only
        test_data = data_obj.get('test', {})

        # Extract 2D, 2D normalized, 3D, and 3D normalized test data
        self.test_2d = test_data.get('2d', None)
        # self.test_2d_normalized = test_data.get('normalized_2d', None)
        self.test_3d = test_data.get('3d', None)
        # self.test_3d_normalized = test_data.get('normalized_3d', None)

        # Print shape information for debugging
        # print(f"Test 2D shape: {self.test_2d.shape if self.test_2d is not None else 'None'}")
        # print(f"Test 2D Normalized shape: {self.test_2d_normalized.shape if self.test_2d_normalized is not None else 'None'}")
        # print(f"Test 3D shape: {self.test_3d.shape if self.test_3d is not None else 'None'}")
        # print(f"Test 3D Normalized shape: {self.test_3d_normalized.shape if self.test_3d_normalized is not None else 'None'}")
        print("Sucessfully loaded GPA dataset.")

    def data(self):
        return self.test_3d, self.test_2d
    



# class H36M(Dataset):
#     def __init__(self, config):
#         self.config = config
#         print('Converting original Human3.6M dataset from', self.config['path_to_dataset'], '(CDF files)')
#         output = {}

#         # Resolve paths for the 3D data directory
#         data_3d_dir = os.path.abspath(self.config.get('data_dir', '.'))
#         os.makedirs(data_3d_dir, exist_ok=True)
        
#         # Define the dynamic output file path for 3D poses
#         output_filename_3d = os.path.join(data_3d_dir, 'data_3d_h36m.npz')
#         # World Coordinates 
#         for subject in subjects:
#             output[subject] = {}
#             file_list = glob(self.config['path_to_dataset'] + '/' + subject + '/MyPoseFeatures/D3_Positions/*.cdf')
#             assert len(file_list) == 30, f"Expected 30 files for subject {subject}, got {len(file_list)}"
            
#             for f in file_list:
#                 action = os.path.splitext(os.path.basename(f))[0]
                
#                 if subject == 'S11' and action == 'Directions':
#                     continue  # Discard corrupted video
                
#                 # Use consistent naming convention
#                 canonical_name = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')
                
#                 hf = cdflib.CDF(f)
#                 positions = hf['Pose'].reshape(-1, 32, 3)
#                 positions /= 1000  # Meters instead of millimeters
#                 output[subject][canonical_name] = positions.astype('float32')

#         print('Saving 3D poses...')
#         np.savez_compressed(output_filename_3d, positions_3d=output)
#         print(f'3D Dataset saved to {output_filename_3d}')

#         # Create 2D pose file
#         print('\nComputing ground-truth 2D poses...')
#         self._dataset = Human36mDataset(output_filename_3d, self.config)
#         output_2d_poses = {}

#         # Resolve paths for the 2D data directory
#         data_2d_dir = os.path.abspath(self.config.get('data_dir', '.'))
#         os.makedirs(data_2d_dir, exist_ok=True)
    
#         # Define the dynamic output file path for 2D poses
#         output_filename_2d = os.path.join(data_2d_dir, 'data_2d_h36m_gt.npz')

#         for subject in self._dataset.subjects():
#             output_2d_poses[subject] = {}
#             for action in self._dataset[subject].keys():
#                 anim = self._dataset[subject][action]
                
#                 positions_2d = []
#                 for cam in anim['cameras']:
#                     pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
#                     pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
#                     pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
#                     positions_2d.append(pos_2d_pixel_space.astype('float32'))
#                 output_2d_poses[subject][action] = positions_2d

#         print('Saving 2D poses...')
#         metadata = {
#             'num_joints': self._dataset.skeleton().num_joints(),
#             'keypoints_symmetry': [self._dataset.skeleton().joints_left(), self._dataset.skeleton().joints_right()]
#         }
#         np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
#         print(f'2D Dataset saved to {output_filename_2d}')
#         print('Done.')

#         # Initialize the internal Human36mDataset
#         self._dataset.preprocess(self.config, output_filename_2d)        
#         self.kps_left = self._dataset.kps_left
#         self.kps_right = self._dataset.kps_right
#         self.joints_left = self._dataset.joints_left
#         self.joints_right = self._dataset.joints_right
#         self.keypoints = self._dataset.keypoints