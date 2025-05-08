# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.utils.data import Dataset


class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, video_mode=False, chunk_size=1):
        self.video_mode = video_mode
        self.chunk_size = chunk_size
        
        # poses_3d is now a single NumPy array of shape (N, joints, 3)
        # poses_2d is now a single NumPy array of shape (N, joints, 2)
        assert poses_3d is not None, "3D poses cannot be None!"
        assert poses_2d is not None, "2D poses cannot be None!"

        # print(poses_3d.shape) # Only prints when using GPA dataset, otherwise h36m dataset there is an error here 
        # print(poses_2d.shape) # Only prints when using GPA dataset, otherwise h36m dataset there is an error here 
        if not self.video_mode:
            # Just store them directly (no further concatenation needed)
            self._poses_3d = poses_3d
            self._poses_2d = poses_2d
        else:
            # If you do want to handle video_mode chunking, do it here:
            all_3d = []
            all_2d = []
            for start_idx in range(0, poses_3d.shape[0], self.chunk_size):
                end_idx = start_idx + self.chunk_size
                all_3d.append(poses_3d[start_idx:end_idx])
                all_2d.append(poses_2d[start_idx:end_idx])
            self._poses_3d = all_3d  # list of arrays
            self._poses_2d = all_2d  # list of arrays

    def __len__(self):
        return len(self._poses_3d)

    def __getitem__(self, index):
        if not self.video_mode:
            pose_3d = torch.from_numpy(self._poses_3d[index]).float()   # (joints, 3)
            pose_2d = torch.from_numpy(self._poses_2d[index]).float()   # (joints, 2)
            return pose_3d, pose_2d
        else:
            # video mode
            pose_3d_seq = torch.from_numpy(self._poses_3d[index]).float()  # chunk_size x (joints, 3)
            pose_2d_seq = torch.from_numpy(self._poses_2d[index]).float()  # chunk_size x (joints, 2)
            return pose_3d_seq, pose_2d_seq

