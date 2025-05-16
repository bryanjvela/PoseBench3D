# import torch
from typing import List, Tuple
import os
import torch
import numpy as np
import pickle
import copy
import cv2
from tqdm import tqdm
import cdflib
import ijson
import matplotlib.pyplot as plt
from poseutils.constants import *
from poseutils import view as v
# from poseutils.view import draw_skeleton
from poseutils.transform import normalize_zscore, normalize_skeleton
from poseutils.props import get_body_centered_axes
import poseutils.camera_utils as cameras
import glob


class DPW3:
    def __init__(self, path: str):
        self.path = path

        self.num_joints = 16
        self._cameras_metadata = []

        self.train_scenes = {}
        self.valid_scenes = {}
        self.test_scenes = {}

        self.load_data()

    def load_data(self):
        """Convert 3DPW into 16 joint format and store 3D, 2D, and camera metadata."""
        augment = False
        train_path = os.path.join(self.path, 'train')
        test_path = os.path.join(self.path, 'test')
        valid_path = os.path.join(self.path, 'validation')
        selected_keys = ["pos_3d", "pos_3d_cam", "frame_ids", "pos_2d"]
        specials = ["outdoors_crosscountry_00"]

        # ---------- TRAIN ----------
        print("Converting training...")
        train_files = glob.glob(train_path + "/*.pkl")
        data_train = {
            "images": [], "size": [],
            "pos_3d_cam": np.zeros((0, 24, 3), dtype=np.float32),
            "pos_2d": np.zeros((0, 24, 2), dtype=np.float32),
            "pos_3d": np.zeros((0, 16, 3), dtype=np.float32),
            "files": [], "fs": [], "cs": [], "camera_metadata": []
        }

        for file in train_files:
            filename = os.path.splitext(os.path.basename(file))[0]
            if filename in specials:
                continue
            data_train[filename] = self.read_data(file, augment=augment)

            data_count = data_train[filename]['combined']['pos_3d_cam'].shape[0]
            data_train["images"].extend([filename] * data_count)
            data_train["pos_3d"] = np.vstack((data_train["pos_3d"], data_train[filename]["combined"]["pos_3d_world"]))
            data_train["pos_3d_cam"] = np.vstack((data_train["pos_3d_cam"], data_train[filename]["combined"]["pos_3d_cam"]))
            data_train["pos_2d"] = np.vstack((data_train["pos_2d"], data_train[filename]["combined"]["pos_2d"]))
            data_train["size"].extend([data_train[filename]["size"]] * data_count)
            data_train["files"].extend(data_train[filename]["combined"]["frame_ids"])
            data_train["fs"].extend(data_train[filename]["combined"]["fs"])
            data_train["cs"].extend(data_train[filename]["combined"]["cs"])
            data_train["camera_metadata"].extend(data_train[filename]["camera_meta"])

            assert len(data_train["files"]) == data_train["pos_2d"].shape[0]
            assert len(data_train["fs"]) == data_train["pos_2d"].shape[0]

        train_scenes = {
            key: {sub_key: value for sub_key, value in data_train[key].items() if sub_key in selected_keys}
            for key in list(data_train.keys())[9:]
        }

        # ---------- VALID ----------
        print("Converting validation...")
        valid_files = glob.glob(valid_path + "/*.pkl")
        data_valid = {
            "images": [], "size": [],
            "pos_3d_cam": np.zeros((0, 24, 3), dtype=np.float32),
            "pos_2d": np.zeros((0, 24, 2), dtype=np.float32),
            "pos_3d": np.zeros((0, 16, 3), dtype=np.float32),
            "files": [], "fs": [], "cs": [], "camera_metadata": []
        }

        for file in valid_files:
            filename = os.path.splitext(os.path.basename(file))[0]
            if filename in specials:
                continue
            data_valid[filename] = self.read_data(file)
            data_count = data_valid[filename]['combined']['pos_3d_cam'].shape[0]
            data_valid["images"].extend([filename] * data_count)
            data_valid["pos_3d"] = np.vstack((data_valid["pos_3d"], data_valid[filename]["combined"]["pos_3d_world"]))
            data_valid["pos_3d_cam"] = np.vstack((data_valid["pos_3d_cam"], data_valid[filename]["combined"]["pos_3d_cam"]))
            data_valid["pos_2d"] = np.vstack((data_valid["pos_2d"], data_valid[filename]["combined"]["pos_2d"]))
            data_valid["size"].extend([data_valid[filename]["size"]] * data_count)
            data_valid["files"].extend(data_valid[filename]["combined"]["frame_ids"])
            data_valid["fs"].extend(data_valid[filename]["combined"]["fs"])
            data_valid["cs"].extend(data_valid[filename]["combined"]["cs"])
            data_valid["camera_metadata"].extend(data_valid[filename]["camera_meta"])

            assert len(data_valid["files"]) == data_valid["pos_2d"].shape[0]
            assert len(data_valid["fs"]) == data_valid["pos_2d"].shape[0]

        valid_scenes = {
            key: {sub_key: value for sub_key, value in data_valid[key].items() if sub_key in selected_keys}
            for key in list(data_valid.keys())[9:]
        }

        # ---------- TEST ----------
        print("Converting test...")
        test_files = glob.glob(test_path + "/*.pkl")
        data_test = {
            "images": [], "size": [],
            "pos_3d_cam": np.zeros((0, 24, 3), dtype=np.float32),
            "pos_2d": np.zeros((0, 24, 2), dtype=np.float32),
            "pos_3d": np.zeros((0, 16, 3), dtype=np.float32),
            "files": [], "fs": [], "cs": [], "camera_metadata": []
        }

        for file in test_files:
            filename = os.path.splitext(os.path.basename(file))[0]
            if filename in specials:
                continue
            data_test[filename] = self.read_data(file)
            data_count = data_test[filename]['combined']['pos_3d_cam'].shape[0]
            data_test["images"].extend([filename] * data_count)
            data_test["pos_3d"] = np.vstack((data_test["pos_3d"], data_test[filename]["combined"]["pos_3d_world"]))
            data_test["pos_3d_cam"] = np.vstack((data_test["pos_3d_cam"], data_test[filename]["combined"]["pos_3d_cam"]))
            data_test["pos_2d"] = np.vstack((data_test["pos_2d"], data_test[filename]["combined"]["pos_2d"]))
            data_test["size"].extend([data_test[filename]["size"]] * data_count)
            data_test["files"].extend(data_test[filename]["combined"]["frame_ids"])
            data_test["fs"].extend(data_test[filename]["combined"]["fs"])
            data_test["cs"].extend(data_test[filename]["combined"]["cs"])
            data_test["camera_metadata"].extend(data_test[filename]["camera_meta"])

            assert len(data_test["files"]) == data_test["pos_2d"].shape[0]
            assert len(data_test["fs"]) == data_test["pos_2d"].shape[0]

        test_scenes = {
            key: {sub_key: value for sub_key, value in data_test[key].items() if sub_key in selected_keys}
            for key in list(data_test.keys())[9:]
        }

        self._3d_world = data_test['pos_3d']
        self._2d = data_test['pos_2d']
        self._cameras_metadata = data_test['camera_metadata']
        self.train_scenes = train_scenes
        self.test_scenes = test_scenes
        self.valid_scenes = valid_scenes

        self.all_data = {
            "train": {"3d": data_train["pos_3d_cam"], "2d": data_train["pos_2d"]},
            "test": {"3d": data_test["pos_3d_cam"], "2d": data_test["pos_2d"]}
        }
        #np.savez_compressed("full_3dpw_test_train_data_gt", data=self.all_data)


    def read_data(self, dataset_path, mul_factor=1, augment=False):
        
        if self.num_joints == 16:
        
            indices_to_select = [0, 2, 5, 8, 1, 4, 7, 6, 12, 15, 16, 18, 20, 17, 19, 21]
            indices_to_sort = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        elif self.num_joints == 17:
            # change select
            indices_to_select = [0, 2, 5, 8, 1, 4, 7, 6, 12, 15, 16, 18, 20, 17, 19, 21]
            indices_to_sort = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        



        print("Dataset_path:", dataset_path)

        data = np.load(dataset_path, allow_pickle=True, encoding='latin1')

        w, h = int(data['cam_intrinsics'][0, 2] * 2), int(data['cam_intrinsics'][1, 2] * 2)

        center = normalize_screen_coordinates(data['cam_intrinsics'][:2, 2], w=w, h=h)
        focus = np.array([data['cam_intrinsics'][0, 0], data['cam_intrinsics'][1, 1]]) / w * 2.0
        intrinsic = np.concatenate((focus, center, [0, 0, 0, 0, 0]))

        data_2d = []
        data_3d_world = []
        data_3d_cam = []
        file_names = []
        pos_angles = []
        fs = []
        cs = []
        cam_poses = {}

        angles = [0, 1./2, 1./3, 1./4, 1./6]

        combined_world_3d = np.zeros((0, 16, 3), dtype=np.float32)
        combined_cam_3d = np.zeros((0, 24, 3), dtype=np.float32)
        combined_proj_2d = np.zeros((0, 24, 2), dtype=np.float32)

        for i in range(len(data['jointPositions'])):
            
            cam_3d = np.zeros((0, 24, 3), dtype=np.float32)
            proj_2d = np.zeros((0, 24, 2), dtype=np.float32)
            world_3d = np.zeros((0, 16, 3), dtype=np.float32)
            current_filenames = []

            for j in tqdm(range(data['jointPositions'][i].shape[0])):

                if data['campose_valid'][i][j] == 0:
                    continue

                if not augment:
                    angles = [0]

                cam_poses[data['sequence'] + "/" + str(data['img_frame_ids'][j])] = data['cam_poses'][j, :, :]

                for angle_i in angles:

                    pos_angles.append(angle_i)

                    pos_3d_world = (data['jointPositions'][i][j, :].reshape((1, 24, 3))*mul_factor)

                    if augment:
                        pos_3d_world_y = rotate_y(pos_3d_world[0, :, :]-pos_3d_world[0, 0, :], np.pi*angle_i)
                        pos_3d_world_y_full = (pos_3d_world_y + pos_3d_world[0, 0, :]).reshape((1, 24, 3))
                        pos_3d_world_y = pos_3d_world_y_full[:, indices_to_select, :]
                        pos_3d_world_y = pos_3d_world_y_full[:, indices_to_sort, :]

                    else:
                        pos_3d_world_y_full = pos_3d_world
                        pos_3d_world_y = pos_3d_world_y_full[:, indices_to_select, :]
                        pos_3d_world_y = pos_3d_world_y_full[:, indices_to_sort, :]

                    data['cam_poses'][j, 2, :3] *= mul_factor

                    pos_3d_cam = np.matmul(data['cam_poses'][j, :, :], np.vstack((pos_3d_world_y_full[0, :, :].T, np.ones((1, 24)))))
                    pos_3d_cam = pos_3d_cam[:3, :].T.reshape((1, 24, 3))
                    pos_2d = wrap(project_to_2d, True, pos_3d_cam, intrinsic)
                    
                    # pos_2d = pos_2d[:, indices_to_select, :]
                    # pos_2d = pos_2d[:, indices_to_sort, :]
                    pos_2d_pixel_space = image_coordinates(pos_2d, w=w, h=h)
                
                    
                    cam_3d = np.vstack((cam_3d, pos_3d_cam))
                    proj_2d = np.vstack((proj_2d, pos_2d_pixel_space))
                    world_3d = np.vstack((world_3d, pos_3d_world_y))
                    fs.append([data['cam_intrinsics'][0, 0], data['cam_intrinsics'][1, 1]])
                    cs.append([data['cam_intrinsics'][0, 2], data['cam_intrinsics'][1, 2]])

                    current_filenames.append(data['sequence'] + "/" + str(data['img_frame_ids'][j]))

                    cam_rotation = data['cam_poses'][j, :3, :3]  # (N, 3, 3)
                    cam_translation = data['cam_poses'][j, :3, 3]  # (N, 3)

                    

                    fc = [data['cam_intrinsics'][0, 0], data['cam_intrinsics'][1, 1]]
                    c = [data['cam_intrinsics'][0, 2], data['cam_intrinsics'][1, 2]]

                    tangential_distortion = np.zeros(2)  # 3DPW does not include tangential coords
                    radial_distortion = np.zeros(3) # 3DPW does not include radial coords

                    camera_metadata = np.concatenate([
                        cam_rotation.flatten(),
                        cam_translation.flatten(),
                        fc,
                        c,
                        tangential_distortion.flatten(),
                        radial_distortion.flatten()
                    ])
                    
                    assert len(camera_metadata) == 21, f"camera_metadata length is {len(camera_metadata)}, expected 21"
                    
                    self._cameras_metadata.append(camera_metadata)

            """
            data_2d.append(proj_2d)
            data_3d_cam.append(cam_3d)
            data_3d_world.append(pos_3d_world_y)

            combined_cam_3d = np.vstack((combined_cam_3d, cam_3d))
            combined_proj_2d = np.vstack((combined_proj_2d, proj_2d))
            combined_world_3d = np.vstack((combined_world_3d, pos_3d_world_y))
            """

            data_2d.append(proj_2d)
            data_3d_cam.append(cam_3d)
            data_3d_world.append(world_3d)
            
            file_names.extend(current_filenames)

        combined_cam_3d = np.vstack(data_3d_cam)
        combined_proj_2d = np.vstack(data_2d)
        combined_world_3d = np.vstack(data_3d_world)

        

        assert combined_proj_2d.shape[0] == len(file_names)
        assert len(data_3d_world) == len(data_3d_cam)
        assert len(fs) == len(file_names)
        assert len(cs) == len(file_names)

        return {
            "size": [w, h],
            "pos_3d": data_3d_world,
            "pos_3d_cam": data_3d_cam,
            "camera_meta": self._cameras_metadata,
            "frame_ids": file_names,
            "combined": {
                "pos_3d_cam": combined_cam_3d,
                "pos_2d": combined_proj_2d,
                "pos_3d_world": combined_world_3d,
                "angles": pos_angles,
                "frame_ids": file_names,
                "fs": fs,
                "cs": cs
                },
            
            "pos_2d": data_2d,
            "cam_poses_by_frame": cam_poses,
            "cam_extrinsics": data['cam_poses'],
            "cam_intrinsics": data['cam_intrinsics']
        }
    
    
def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / (w * 1.0)]



def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, (h * 1.0) / w]) * (w * 1.0) / 2



    


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                        keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c




class GPA_Module():
    
  def __init__(self, path):
        self.path = path
      
        self.data_2d_train = []
        self.data_3d_train = []
        self.data_2d_test = []
        self.data_3d_test = []
        self.data_2d_c_train = []
        self.data_2d_c_test = []
        self.data_cam_center_train = []
        self.data_cam_focal_train = []
        self.data_cam_center_test = []
        self.data_cam_focal_test = []
        self.data_bbox_train = []
        self.data_bbox_test = []
        self.num_joints = 16 #TODO: change to 14

        self.load_data()

    
  def load_data(self):
        if self.num_joints == 16:
            
            indices_to_select = [0, 24, 25, 26, 29, 30, 31, 2, 5, 6, 7, 17, 18, 19, 9, 10, 11]
            indices_to_sort = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]

        with open(self.path, 'r') as file:

            json_data = ijson.items(file, 'annotations.item')

            for entry in tqdm(json_data, total=304892):
                joint_2d = np.array(entry['joint_imgs_uncrop'], dtype=np.float32).reshape((1, 34, 2))
                joint_2d_c = np.array(entry['joint_imgs'], dtype=np.float32).reshape((1, 34, 2))
                joint_3d = np.array(entry['joint_cams'], dtype=np.float32).T.reshape((1, 34, 3))
                cam_center = np.array(entry['c'], dtype=np.float32).reshape((1, 2))
                cam_focal = np.array(entry['f'], dtype=np.float32).reshape((1, 2))
                bbox = np.array(entry['bboxes'], dtype=np.float32)

                if entry['istrains']:
                    self.data_2d_train.append(joint_2d)
                    self.data_2d_c_train.append(joint_2d_c)
                    self.data_3d_train.append(joint_3d)
                    self.data_cam_center_train.append(cam_center)
                    self.data_cam_focal_train.append(cam_focal)
                    self.data_bbox_train.append(bbox)
                elif entry['istests']:
                    self.data_2d_test.append(joint_2d)
                    self.data_2d_c_test.append(joint_2d_c)
                    self.data_3d_test.append(joint_3d)
                    self.data_cam_center_test.append(cam_center)
                    self.data_cam_focal_test.append(cam_focal)
                    self.data_bbox_test.append(bbox)

            file.close()

            #print(np.vstack(data_2d_train).shape)

            data = {
                "size": [1920, 1080],
                "size_cropped": [256, 256],
                "train": {
                    "2d": np.vstack(self.data_2d_train),
                    "2d_c": np.vstack(self.data_2d_c_train),
                    "3d": np.vstack(self.data_3d_train),
                    "center": np.vstack(self.data_cam_center_train),
                    "focus": np.vstack(self.data_cam_focal_train),
                    "bbox": np.vstack(self.data_bbox_train)
                },
                "test": {
                    "2d": np.vstack(self.data_2d_test),
                    "2d_c": np.vstack(self.data_2d_c_test),
                    "3d": np.vstack(self.data_3d_test),
                    "center": np.vstack(self.data_cam_center_test),
                    "focus": np.vstack(self.data_cam_focal_test),
                    "bbox": np.vstack(self.data_bbox_test)
                }   
            }
                
        # self.all_data = {
        #             "train": {"3d": data["train"]["3d_joint"], "2d": data["train"]["2d_joint"]},
        #             "test": {"3d": data["test"]["3d_joint"], "2d": data["test"]["2d_joint"]}
        #         }
        self.all_data = data
            #np.savez_compressed("conformed_GPA_meters1", data=data)
