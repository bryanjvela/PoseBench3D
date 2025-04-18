# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np
from . import tools
import torch
from torch.utils.data import Dataset
from functools import reduce


# class PoseGenerator(Dataset):
#     def __init__(self, poses_3d, poses_2d, actions, video_mode=False, chunk_size=1):
#         """
#         poses_3d: list of arrays, each shape (L_i, J, 3)
#         poses_2d: list of arrays, each shape (L_i, J, 2)
#         actions:  list of lists of action labels
#         video_mode: if True, we do chunking
#         chunk_size: if 1, each sub-sequence is length 1
#         """
#         self.video_mode = video_mode
#         self.chunk_size = chunk_size
#         assert poses_3d is not None, "3D poses cannot be None!"
#         #print(actions)
#         #print(type(actions))
#         if not video_mode:
#             # Non-video flattening (original approach)
#             self._poses_3d = np.concatenate(poses_3d, axis=0)  # shape (N, J, 3)
#             self._poses_2d = np.concatenate(poses_2d, axis=0)  # shape (N, J, 2)

#             # Handle single-action case
#             if isinstance(actions, str):
#                 # Single string => replicate for all frames
#                 single_label = actions
#                 self._actions = [single_label] * self._poses_3d.shape[0]
#             else:
#                 #print('xxxxxxxxx')
#                 self._actions = reduce(lambda x, y: x + y, actions)

#             #self._actions = reduce(lambda x, y: x + y, actions)

#             #print(self._poses_3d.shape[0])
#             #print(len(self._actions))
#             assert self._poses_3d.shape[0] == self._poses_2d.shape[0]
#             assert self._poses_3d.shape[0] == len(self._actions)
        
#         else:
#             # Video mode: chunk each sequence into sub-sequences of chunk_size
#             # e.g. if chunk_size=1 and a sequence is length L_i, we produce L_i items of shape (1, J, 3).
            
#             all_3d = []
#             all_2d = []
#             all_actions = []

#             for seq_3d, seq_2d, seq_actions in zip(poses_3d, poses_2d, actions):
#                 L = seq_3d.shape[0]

#                 for start_idx in range(0, L, self.chunk_size):
#                     end_idx = start_idx + self.chunk_size

#                     chunk_3d = seq_3d[start_idx:end_idx]  
#                     chunk_2d = seq_2d[start_idx:end_idx]

#                     chunk_actions = seq_actions  
                    
#                     all_3d.append(chunk_3d)
#                     all_2d.append(chunk_2d)
#                     all_actions.append(chunk_actions)

#             self._poses_3d = all_3d
#             self._poses_2d = all_2d
#             self._actions = all_actions

#     def __len__(self):
#         return len(self._poses_3d)

#     def __getitem__(self, index):
#         if not self.video_mode:
#             # Non-video (original flatten)
#             out_pose_3d = torch.from_numpy(self._poses_3d[index]).float()  # shape (J,3)
#             out_pose_2d = torch.from_numpy(self._poses_2d[index]).float()  # shape (J,2)
#             out_action = self._actions[index]
#             return out_pose_3d, out_pose_2d, out_action
#         else:
#             # Video mode
#             # each item is a small chunk of shape (chunk_size, J, 3) or (chunk_size, J, 2)
#             pose_3d_seq = torch.from_numpy(self._poses_3d[index]).float()  # (chunk_size, J, 3)
#             pose_2d_seq = torch.from_numpy(self._poses_2d[index]).float()  # (chunk_size, J, 2)
#             action_label = self._actions[index]  # e.g. shape (chunk_size,) if you stored full sub-range
#             return pose_3d_seq, pose_2d_seq, action_label

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





# class PoseGenerator(Dataset):
#     def __init__(self, poses_3d, poses_2d, actions):
#         assert poses_3d is not None
#         self._poses_3d = np.concatenate(poses_3d)
#         self._poses_2d = np.concatenate(poses_2d)
#         self._actions = reduce(lambda x, y: x + y, actions)

#         # In your logs, you see shape (N, 17, 2). Let's confirm or debug-print here:
#         print("Concatenated poses_2d shape:", self._poses_2d.shape)
        
#         # If the model expects only 16, we can remove joint index 0 in the entire array right away:
#         # E.g., remove the 'root' at index 0:
#         self._poses_2d = self._poses_2d[:, 1:, :]  # new shape: (N, 16, 2)

#         print("After slicing, poses_2d shape:", self._poses_2d.shape)
#         print('Generating {} poses...'.format(len(self._actions)))

#     def __getitem__(self, index):
#         out_pose_3d = torch.from_numpy(self._poses_3d[index]).float()
#         out_pose_2d = torch.from_numpy(self._poses_2d[index]).float()
#         out_action = self._actions[index]

#         return out_pose_3d, out_pose_2d, out_action

#     def __len__(self):
#         return len(self._actions)


class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.

    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, noisy=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2*pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.noisy = noisy
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift

                    # 2D poses
                    seq_2d = self.poses_2d[seq_i]
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]

                    if flip:
                        # Flip 2D keypoints
                        # self.batch_2d = np.flip(self.batch_2d, 1)
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        low_3d = max(start_3d, 0)
                        high_3d = min(end_3d, seq_3d.shape[0])
                        pad_left_3d = low_3d - start_3d
                        pad_right_3d = end_3d - high_3d
                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            # Flip 3D joints
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                    self.batch_3d[i, :, self.joints_right + self.joints_left]

                    # Cameras
                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            # Flip horizontal distortion coefficients
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, self.batch_2d[:len(chunks)]
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
                else:
                    yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]

            if self.endless:
                self.state = None
            else:
                enabled = False

class Augmented_Train_ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.

    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, noisy=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2*pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.noisy = noisy
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift

                    # 2D poses
                    seq_2d = self.poses_2d[seq_i]
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]

                    if flip:
                        # Flip 2D keypoints
                        # self.batch_2d = np.flip(self.batch_2d, 1)
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        low_3d = max(start_3d, 0)
                        high_3d = min(end_3d, seq_3d.shape[0])
                        pad_left_3d = low_3d - start_3d
                        pad_right_3d = end_3d - high_3d
                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            # Flip 3D joints
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                    self.batch_3d[i, :, self.joints_right + self.joints_left]

                        random_move=True
                        if random_move: # bruce
                            # batch_3d: torch.Size([64, 1, 17, 3])
                            b_3d = torch.from_numpy(self.batch_3d[i, :, :, :]).float()
                            b_3d = b_3d.permute(2,0,1).view(3,1,17,1)
                            b_3d = tools.random_move(b_3d.numpy())#.view(3,1,17,1)
                            b_3d = torch.from_numpy(b_3d).float()
                            b_3d = b_3d.view(3,1,17).permute(1,2,0)
                            self.batch_3d[i, :, :, :] = b_3d.numpy()

                    # Cameras
                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            # Flip horizontal distortion coefficients
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, self.batch_2d[:len(chunks)]
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
                else:
                    yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]

            if self.endless:
                self.state = None
            else:
                enabled = False

class Evaluate_Generator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.

    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
        
        # Build lineage info
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
        
        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        
        if augment:
            self.batch_2d_flip = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))
            self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))
        else:
            self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))
            #print("generator batch 2d shape: ", self.batch_2d.shape)

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i * self.batch_size: (b_i + 1) * self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift

                    # 2D poses
                    seq_2d = self.poses_2d[seq_i]
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        #print("enter 1")
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                                                  'edge')
                        if self.augment:
                            self.batch_2d_flip[i] = np.pad(seq_2d[low_2d:high_2d],
                                                      ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                                                      'edge')

                    else:
                        #print("enter 2")
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]
                        if self.augment:
                            self.batch_2d_flip[i] = seq_2d[low_2d:high_2d]

                    if self.augment:
                        self.batch_2d_flip[i, :, :, 0] *= -1
                        self.batch_2d_flip[i, :, self.kps_left + self.kps_right] = self.batch_2d_flip[i, :, self.kps_right + self.kps_left]

                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        low_3d = max(start_3d, 0)
                        high_3d = min(end_3d, seq_3d.shape[0])
                        pad_left_3d = low_3d - start_3d
                        pad_right_3d = end_3d - high_3d
                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d],
                                                      ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                self.batch_3d[i, :, self.joints_right + self.joints_left]

                    # Cameras
                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            # Flip horizontal distortion coefficients
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)

                if self.augment:
                    if self.poses_3d is None and self.cameras is None:
                        yield None, None, self.batch_2d[:len(chunks)], self.batch_2d_flip[:len(chunks)]
                    elif self.poses_3d is not None and self.cameras is None:
                        yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)], self.batch_2d_flip[:len(chunks)]
                    elif self.poses_3d is None:
                        yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)], self.batch_2d_flip[:len(chunks)]
                    else:
                        yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)], self.batch_2d_flip[:len(chunks)]
                else:
                    if self.poses_3d is None and self.cameras is None:
                        yield None, None, self.batch_2d[:len(chunks)]
                    elif self.poses_3d is not None and self.cameras is None:
                        yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                    elif self.poses_3d is None:
                        yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
                    else:
                        yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]

            if self.endless:
                self.state = None
            else:
                enabled = False
