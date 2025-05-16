from typing import List, Tuple

import os.path
import numpy as np
import copy

import cdflib
import scipy.io

from tqdm import tqdm

from matplotlib import pyplot as plt
# import poseutils
from poseutils.view import *
from poseutils.transform import normalize_zscore
from poseutils.props import calculate_avg_limb_lengths

from smpl_relations import get_intrinsic, get_extrinsic, project_vertices

# from h36m_helpers import data_utils as d
# from h36m_helpers import camera_utils as c

DUMMY_SIZE = 128 

# Design choice: give all data or option to index
# Probably prefer option to index, because extend torch Dataset for mass operation

# Enchancement: Provide abstraction for sections—a recursive data structure
# section_id, section_length
# recursive function converts section_id to a indices

# Root->
    # Subj 0
        # Act 0
            # Frame 0 — 0
            # ... — 1
            # ... — 2
        # Act 1
            # ... 
    # Subj 1
        # Act 1
        # Act 2
        # Act 3

class Dataset:
    # units will be mm
    def __init__(self):
        # list of processed 16-joint skeletons
        self._3d_world = []
        # list of gt 2d points in image
        self._2d = []
        # rotation matrix in global reference, translation vector in global reference, focal length x/y, center x/y, distortion tangential, distortion radial (length 21)
        # CONSIDER: Not flattened
        self._cameras_metadata = [] # array of arrays of len 21
        # cached list of _3d_cam when computed
        self._3d_cam = []
    def load_data(self):
        # Needs to be overridden to initialize _3d_world, _2d, and _cameras_metadata
        pass
    def get_3d_world(self):
        return self._3d_world
    def get_3d_cam(self):
        return self._3d_cam
        # if self._3d_cam is None:
        #     def apply_cam(index):
        #         joint_array = self._3d_world[index]
        #         camera_metadata = self._cameras_metadata[index]
        #         rot_matrix = np.matrix([camera_metadata[0:3], camera_metadata[3:6], camera_metadata[6:9]])
        #         trans_vector = camera_metadata[9:12][:, np.newaxis]
        #         return (rot_matrix @ joint_array.T - trans_vector).T
        #         # return (self.camera_metadata[0] @ joint_array.T - self.camera_metadata[1]).T
        #     self._3d_cam = np.empty(self._3d_world.shape)
        #     for i in range(self._3d_world.shape[0]):
        #         self._3d_cam[i] = apply_cam(i)
        # return self._3d_cam
    def get_cams(self):
        return self._cameras_metadata
    def get_2d(self):
        return self._2d
    def get_images(self):
        # TODO: needed later on
        # discuss with Saad on format
        # could be list of paths
        # might be loaded with image library
        pass



class Surreal(Dataset):
    def load_data(self):
        base_dir = '../SURREAL/data/cmu/train/'
        
        self._3d_cam = []
        self._2d = []
        
        runs = ['run0/', 'run1/', 'run2/']
        
        for run in tqdm(runs, desc="Runs"):
            run_path = os.path.join(base_dir, run)
            dirs = os.listdir(run_path)
            
            for d in tqdm(dirs, desc=f"Inside {run}", leave=False):
                dp = os.path.join(run_path, d)
                if not os.path.isdir(dp):
                    continue

                for d2 in os.listdir(dp):
                    d2p = os.path.join(dp, d2)
                    if d2p.endswith("_info.mat"):
                        # Load the MAT file
                        data = scipy.io.loadmat(d2p)
                        frame_num = data['gender'].shape[0]

                        if frame_num >= 2:
                            # Extract 3D joints
                            joints3D = data['joints3D'].transpose(2, 1, 0)
                            # Copy for camera transform
                            joints3Dcam = joints3D.copy()
                            
                            # Intrinsics / extrinsics
                            intrinsic = get_intrinsic()
                            extrinsic = get_extrinsic(data['camLoc'])[0]

                            for i in range(frame_num):
                                points = joints3Dcam[i]
                                homo_coords = np.concatenate(
                                    [points, np.ones((points.shape[0], 1))],
                                    axis=1
                                ).T
                                
                                # project to 2D
                                proj_coords = intrinsic @ (extrinsic @ homo_coords)
                                proj_coords = proj_coords.T
                                proj_2d = proj_coords[:, :2] / proj_coords[:, -1:]
                                self._2d.append(proj_2d[np.newaxis, :, :])

                                # transform the 3D points (camera coords)
                                joints3Dcam[i] = (extrinsic @ homo_coords).T
                            
                            self._3d_cam.append(joints3Dcam)
        
        # Stack everything once outside the loops
        self._3d_cam = np.vstack(self._3d_cam)  # shape: (N, 24, 3)
        self._2d = np.vstack(self._2d)          # shape: (N, 24, 2)

        # Save
        np.savez_compressed(
            "data/surreal_train_compiled_final", 
            data_3d=self._3d_cam * 1000,  # mm
            data_2d=self._2d
        )


if __name__ == "__main__":
    dset = Surreal()
    dset.load_data()



