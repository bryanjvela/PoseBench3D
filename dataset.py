import os
import sys
sys.path.append('../')
import copy
import numpy as np
import cdflib
from glob import glob
import common.camera as camera
from common.h36m_dataset import h36m_cameras_extrinsic_params, h36m_cameras_intrinsic_params
from common.utils import normalize_screen_coordinates, normalize_data
from poseutils.constants import dataset_indices


class Dataset:
    def data(self):# -> Tuple[List[torch.float32], List[torch.float32]]:
        pass
    

class H36M(Dataset):

    def __init__(self, path, config):
        super(H36M, self).__init__()

        self.config = config
        self.cameras = None
        self._data_train = {"2d": np.zeros((0, 16, 2), dtype=np.float32),
                            "3d": np.zeros((0, 16, 3), dtype=np.float32),
                            "fs": [],
                            "cs": []}
        self._data_valid = {"2d": np.zeros((0, 16, 2), dtype=np.float32),
                            "3d": np.zeros((0, 16, 3), dtype=np.float32),
                            "fs": [],
                            "cs": []}

        self.mean_2d = 0.0
        self.std_2d = 0.0
        self.mean_3d = 0.0
        self.std_3d = 0.0


        self.cameras = []
        self.load_data(path, self.config)

    def load_data(self, path, config):
        print("Path: ", path)

        self.input_shape = config['input_shape']
        self.indices_to_select_2d, self.indices_to_select_3d = dataset_indices(config['dataset'], self.input_shape['num_joints']) # [0, 6, 7, 8, 1, 2, 3, 12, 13, 15, 25, 26, 27, 17, 18, 19]
        # indices_to_select_3d = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]


        self.cameras, self._cameras = self.prepare_cameras(os.path.join(path, "metadata.xml"))

        TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
        TEST_SUBJECTS  = [9, 11]

        actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

        cache_path = os.path.join(path, 'h36m_raw_data.npz')
        if os.path.exists(cache_path):
            print(f"Loading cached datasets from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            trainset = data['trainset'].item()
            testset = data['testset'].item()
        else:
            print("Building datasets from scratch...")
            trainset = self.load_3d_data(path, TRAIN_SUBJECTS, actions)
            testset = self.load_3d_data(path, TEST_SUBJECTS, actions)
            print(f"Saving datasets to {cache_path}")
            np.savez_compressed(cache_path, trainset=trainset, testset=testset)

        # trainset = self.load_3d_data(path, TRAIN_SUBJECTS, actions)
        # testset = self.load_3d_data(path, TEST_SUBJECTS, actions)

        self.model_info = config['model_info']
        d2d_train, rects_train, d3d_train, fs_t, cs_t = self.project_to_cameras(trainset)
        d2d_valid, rects_valid, d3d_valid, fs_v, cs_v = self.project_to_cameras(testset, self.model_info['normalize_2d_to_minus_one_to_one'])

        if self.model_info['output_3d'] == 'millimeters':
            d3d_valid /= 1000.0  # convert from mm to meters
        



        d2d_train = d2d_train[:, self.indices_to_select_2d, :]
        d2d_valid = d2d_valid[:, self.indices_to_select_2d, :]

        self._data_train['2d'] = self.root_center(np.array(d2d_train))
        self._data_valid['2d'] = self.root_center(np.array(d2d_valid), self.model_info['root_center_2d_test_input'])

        self._data_train['3d'] = self.root_center(np.array(d3d_train))[:, self.indices_to_select_2d, :]
        self._data_valid['3d'] = self.root_center(np.array(d3d_valid))[:, self.indices_to_select_2d, :] 
            
        self.mean_3d = np.mean(self._data_train['3d'], axis=0)
        self.std_3d = np.std(self._data_train['3d'], axis=0)
        self.mean_2d = np.mean(self._data_train['2d'], axis=0)
        self.std_2d = np.std(self._data_train['2d'], axis=0)

        if self.model_info['trained_on_normalized_data']:
            self._data_valid['3d'] = normalize_data(self._data_valid['3d'], self.mean_3d, self.std_3d, skip_root=True)
            self._data_valid['2d'] = normalize_data(self._data_valid['2d'], self.mean_2d, self.std_2d, skip_root=True)



    
    def prepare_cameras(self, metadata_path):
        """
        Load and prepare cameras from metadata for Human3.6M.

        Args:
            metadata_path (str): Path to metadata.xml file.

        Returns:
            list: List of prepared camera dictionaries.
        """
        # 1. Load basic camera metadata
        cameras_loaded = camera.load_cameras(metadata_path)

        # 2. Deepcopy base extrinsics
        prepared_cameras = copy.deepcopy(h36m_cameras_extrinsic_params)

        # 3. Update intrinsic + extrinsic parameters
        for cameras in prepared_cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])

                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype=np.float32)

                # Normalize center and scale focal lengths
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype(np.float32)
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2

                # Convert translation to meters
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000.0  # mm to meters

                # Create intrinsic parameter vector
                cam['intrinsic'] = np.concatenate([
                    cam['focal_length'],
                    cam['center'],
                    cam['radial_distortion'],
                    cam['tangential_distortion']
                ])

        return cameras_loaded, prepared_cameras

    def data(self):
        if self.model_info['flattened_coordinate_model']:
            self._data_valid['3d'] = self._data_valid['3d'].reshape((-1, self.input_shape['num_joints']*3))
            self._data_valid['2d'] = self._data_valid['2d'].reshape((-1, self.input_shape['num_joints']*2))
        return self._data_valid['3d'], self._data_valid['2d']

    def load_3d_data(self, bpath, subjects, actions, dim=3):
        if dim not in [2, 3]:
            raise ValueError("dim must be 2 or 3")

        data = {}
        pose_counter = 0  

        for subj in subjects:
            folder_subj = f"S{subj}"

            cdf_pattern = os.path.join(bpath, folder_subj, f"MyPoseFeatures/D{dim}_Positions", "*.cdf")
            file_list = glob(cdf_pattern)

            for f in file_list:
                seqname = os.path.splitext(os.path.basename(f))[0]  # e.g. 'Directions 1'
            

            
                if subj == 11 and seqname == "Directions":
                    continue  # Discard corrupted video

                action_raw = seqname
                canonical_name = (action_raw
                                .replace('TakingPhoto', 'Photo')
                                .replace('WalkingDog', 'WalkDog'))

                # Load the .cdf
                cdf_file = cdflib.CDF(f)
                poses = cdf_file.varget("Pose").squeeze()  # shape: (N*32*3)
                poses = poses.reshape((-1, 32, 3))         # shape: (N, 32, 3)

                # If subject is 9 or 11, count frames
                pose_counter += poses.shape[0]

                # Store in the dictionary, keyed by (subj, canonical_name, seqname)
                data[(subj, canonical_name, seqname)] = poses

        if 9 in subjects or 11 in subjects:
            print(f"Total number of test poses (frames): {pose_counter}")
        else:
            print(f"Total number of train poses (frames): {pose_counter}")

        return data
   
    def root_center(self, data3d, apply_root_centering=True):
        if apply_root_centering:
            for i in range(data3d.shape[0]):
                data3d[i, :, :] -= data3d[i, 0, :]
        return data3d

    def normalization_stats_3d(self, pose_set_3d):

        for key in pose_set_3d.keys():
            poses = pose_set_3d[key]
            
            for i in range(poses.shape[0]):
                poses[i, :, :] -= poses[i, 0, :]

            pose_set_3d[key] = poses

        complete_data = np.vstack(pose_set_3d.values())

        return np.mean(complete_data, axis=0), np.std(complete_data, axis=0)

    def project_to_cameras( self, poses_set, normalize_2d=False):
        """
        Project 3d poses using camera parameters

        Args
        poses_set: dictionary with 3d poses
        cams: dictionary with camera parameters
        ncams: number of cameras per subject
        Returns
        t2d: dictionary with 2d poses
        """
        t2d = []
        rects = []
        t3d = []
        fs = []
        cs = []

        half_diag = np.sqrt(2)/2
        rect_points = np.array([[half_diag, half_diag],[half_diag, -half_diag],[-half_diag, -half_diag],[-half_diag, half_diag]])
        rect_norm = np.linalg.norm(rect_points, axis=1)
        rect_points /= rect_norm[:, np.newaxis]
        rect_3d_pts = np.zeros((4, 3))
        rect_3d_pts[:, :2] = rect_points

        total_points = 0
        once = False
        # print(poses_set.keys())
        for key in poses_set.keys():
            (subj, action, sqename) = key
            t3dw = poses_set[key]
            # print(type(t3dw))
            t3dw_rect = np.repeat(rect_3d_pts.reshape(-1, 12)*200, t3dw.shape[0], axis=0)
            t3dw_rect = t3dw[:, :1, :] + t3dw_rect.reshape(-1, 4, 3)

            z_max, z_min = np.max(t3dw[0, :, 2]), np.min(t3dw[0, :, 2])
            ground_offset = abs(z_min-z_max)*0.6
            t3dw_rect[:, :, 2] -= ground_offset

            for cam in range(4):
                R, T, f, c, k, p, name = self.cameras[ (subj, cam+1) ]
                # print(self.cameras[(subj, cam+1)])
                t3dc = camera.world_to_camera_frame( np.reshape(t3dw, [-1, 3]), R, T)
                pts2d, _, _, _, _ = camera.project_point_radial( np.reshape(t3dw, [-1, 3]), R, T, f, c, k, p )
                # print(type(pts2d))
                # print(pts2d.shape)
                rect2d, _, _, _, _ = camera.project_point_radial( np.reshape(t3dw_rect, [-1, 3]), R, T, f, c, k, p)
                pts2d = np.reshape( pts2d, [-1, 32, 2] )

                if subj in [9, 11] and normalize_2d:
                    # We need a string key like 'S9' or 'S11' if that's how _cameras is indexed
                    subject_str = f"S{subj}"
                    cam_data = self._cameras[subject_str][cam]  # 0..3
                    pts2d = normalize_screen_coordinates(pts2d,
                                                        w=cam_data['res_w'],
                                                        h=cam_data['res_h'])
                # exit(0)
                total_points += pts2d.shape[0]
                t2d.append(pts2d)
                t3d.append(t3dc.reshape((-1, 32, 3)))
                rects.append(rect2d.reshape((-1, 4, 2)))
                fs.append([f.T]*pts2d.shape[0])
                cs.append([c.T]*pts2d.shape[0])

        t2d = np.vstack(t2d)
        rects = np.vstack(rects)
        t3d = np.vstack(t3d)
        fs = np.vstack(fs)
        cs = np.vstack(cs)

        print("Projected points: ", total_points)

        return t2d, rects, t3d, fs, cs



class GPA(Dataset):

    def __init__(self, path, config):
        
        super(GPA, self).__init__()
        self.cameras = None
        self.config = config

        self._data_train = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}
        self._data_valid = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}

        self.mean_2d = 0.0
        self.std_2d = 0.0
        self.mean_3d = 0.0
        self.std_3d = 0.0


        self.load_data(path, config)

    def load_data(self, path, config):

        data = np.load(path, allow_pickle=True, encoding='latin1')['data'].item()

        data_train = data['train']
        data_valid = data['test']

        self.model_info = config['model_info']
        # indices_to_select = [0, 24, 25, 26, 29, 30, 31, 2, 5, 6, 7, 17, 18, 19, 9, 10, 11]
        # indices_to_sort = [0, 4, 5, 6, 1, 2, 3, 7, 8, 10, 14, 15, 16, 11, 12, 13]
        self.input_shape = config['input_shape']
        to_select, to_sort = dataset_indices(config['dataset'], self.input_shape['num_joints'])


        self._data_train['2d'] = data_train["2d"].reshape(-1,34,2)[:, to_select,  :][:, to_sort, :]
        self._data_train['3d'] = data_train["3d"].reshape(-1,34,3)[:, to_select,  :][:, to_sort, :]

        self._data_valid['2d'] = data_valid["2d"].reshape(-1,34,2)[:, to_select,  :][:, to_sort, :]
        self._data_valid['3d'] = data_valid["3d"].reshape(-1,34,3)[:, to_select,  :][:, to_sort, :]

        if self.model_info['normalize_2d_to_minus_one_to_one']:
            self._data_valid['2d'] = normalize_screen_coordinates(self._data_valid['2d'], w=1920, h=1080)

        if self.model_info['output_3d'] == 'millimeters':
            self._data_valid['3d'] /= 1000.0  # convert from mm to meters

        self._data_train['2d'] = self.root_center_vectorized(np.array(self._data_train['2d']))
        self._data_valid['2d'] = self.root_center_vectorized(np.array(self._data_valid['2d']), self.model_info['root_center_2d_test_input'])

        self._data_train['3d'] = self.root_center_vectorized(np.array(self._data_train['3d']))
        self._data_valid['3d'] = self.root_center_vectorized(np.array(self._data_valid['3d']))


        self.mean_2d = np.mean(self._data_train['2d'], axis=0)
        self.std_2d = np.std(self._data_train['2d'], axis=0)
        self.mean_3d = np.mean(self._data_train['3d'], axis=0)
        self.std_3d = np.std(self._data_train['3d'], axis=0)

        if self.model_info['trained_on_normalized_data']:
                self._data_valid['3d'] = normalize_data(self._data_valid['3d'], self.mean_3d, self.std_3d, skip_root=True)
                self._data_valid['2d'] = normalize_data(self._data_valid['2d'], self.mean_2d, self.std_2d, skip_root=True)

        

    def root_center_vectorized(self, data, apply_root_centering=True):
        """
        Subtract the coordinates of the first joint from all joints in each sample,
        effectively placing the root joint at the origin (0,0,0).

        data shape is typically (N, J, D):
        N = number of samples/frames
        J = number of joints
        D = dimensionality (2 for 2D, 3 for 3D)

        This function replicates the behavior:
        data[i, :, :] -= data[i, 0, :]
        but vectorized to avoid an explicit loop.

        :param data: np.ndarray of shape (N, J, D)
        :param apply_root_centering: whether to apply the root-centering
        :return: root-centered data if apply_root_centering is True,
                else returns data unchanged
        """
        if apply_root_centering:
            # `data[:, :1, :]` extracts just the first joint (shape: [N, 1, D]),
            # and broadcasting subtracts it from all joints in each sample.
            data = data - data[:, :1, :]
        return data
        
    def data(self):
        if self.model_info['flattened_coordinate_model']:
            self._data_valid['3d'] = self._data_valid['3d'].reshape((-1, self.input_shape['num_joints']*3))
            self._data_valid['2d'] = self._data_valid['2d'].reshape((-1, self.input_shape['num_joints']*2))
        return self._data_valid['3d'], self._data_valid['2d']



class Surreal(Dataset):

    def __init__(self, path_train, path_valid, config):
        super(Surreal, self).__init__()

        self.cameras = None

        self.config = config

        self._data_train = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}
        self._data_valid = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}

        self.mean_2d = 0.0
        self.std_2d = 0.0
        self.mean_3d = 0.0
        self.std_3d = 0.0

        self.load_data(path_train, path_valid, config)

    def load_data(self, path_train, path_valid, config):

        data_train = np.load(path_train, allow_pickle=True)
        data_valid = np.load(path_valid, allow_pickle=True)

        # indices_to_arrange = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
        # indices_to_select = [0, 1, 4, 7, 2, 5, 8, 6, 12, 15, 17, 19, 21, 16, 18, 20]

        self.input_shape = config['input_shape']
        self.model_info = config['model_info']
        to_select, to_sort = dataset_indices(config['dataset'], self.input_shape['num_joints'])

        # random_idx = np.random.choice(data_train["data_3d"].shape[0], data_train["data_3d"].shape[0]//6)

        max_idx = data_train["data_3d"].shape[0]//6

        self._data_train['2d'] = data_train['data_2d'] # [random_idx, :, :]
        self._data_train['3d'] = data_train['data_3d'] # [random_idx, :, :]

        i = 3

        self._data_train['2d'] = self._data_train['2d'][(i-2)*max_idx:i*max_idx, to_select, :][:, to_sort,  :]
        self._data_train['3d'] = self._data_train['3d'][(i-2)*max_idx:i*max_idx, to_select, :][:, to_sort,  :]

        self._data_valid['2d'] = data_valid["data_2d"][:, to_select, :][:, to_sort,  :]
        self._data_valid['3d'] = data_valid["data_3d"][:, to_select, :][:, to_sort,  :]

        if self.model_info['normalize_2d_to_minus_one_to_one']:
            self._data_valid['2d'] = normalize_screen_coordinates(self._data_valid['2d'], w=320, h=240)

        if self.model_info['output_3d'] == 'millimeters':
            self._data_valid['3d'] /= 1000.0  # convert from mm to meters


        self._data_train['2d'] = self.root_center_vectorized(np.array(self._data_train['2d']))
        self._data_valid['2d'] = self.root_center_vectorized(np.array(self._data_valid['2d']), self.model_info['root_center_2d_test_input'])

        self._data_train['3d'] = self.root_center_vectorized(np.array(self._data_train['3d']))
        self._data_valid['3d'] = self.root_center_vectorized(np.array(self._data_valid['3d']))


        self.mean_2d = np.mean(self._data_train['2d'], axis=0)
        self.std_2d = np.std(self._data_train['2d'], axis=0)
        self.mean_3d = np.mean(self._data_train['3d'], axis=0)
        self.std_3d = np.std(self._data_train['3d'], axis=0)

    
        if self.model_info['trained_on_normalized_data']:
                self._data_valid['3d'] = normalize_data(self._data_valid['3d'], self.mean_3d, self.std_3d, skip_root=True)
                self._data_valid['2d'] = normalize_data(self._data_valid['2d'], self.mean_2d, self.std_2d, skip_root=True)


    def root_center_vectorized(self, data, apply_root_centering=True):
        if apply_root_centering:
            data = data - data[:, :1, :]
        return data
    
    def data(self):
        if self.model_info['flattened_coordinate_model']:
            self._data_valid['3d'] = self._data_valid['3d'].reshape((-1, self.input_shape['num_joints']*3))
            self._data_valid['2d'] = self._data_valid['2d'].reshape((-1, self.input_shape['num_joints']*2))

        corrupted_64 = [2587, 4164, 12077, 16326]        # original bad batches
        first_bad = [idx * 64 for idx in corrupted_64]   # starting sample indices

        corrupted_samples = []
        for start in first_bad:                          # e.g. 165568, 266496, …
            corrupted_samples.extend(range(start, start + 64))

        corrupted_samples = set(corrupted_samples)       # faster membership tests

        total_samples = self._data_valid['3d'].shape[0]
        keep_mask = np.ones(total_samples, dtype=bool)
        for idx in corrupted_samples:
            if idx < total_samples:                      # guard against overflow
                keep_mask[idx] = False

        clean_3d = self._data_valid['3d'][keep_mask]
        clean_2d = self._data_valid['2d'][keep_mask]
        print(f"Removed {len(corrupted_samples)} corrupted samples "f"({100*len(corrupted_samples)/total_samples:.4f}% of data).")
        return clean_3d, clean_2d
        #return self._data_valid['3d'], self._data_valid['2d']
    


class ThreeDPW(Dataset):

    def __init__(self, path, config):
        super(ThreeDPW, self).__init__()

        self.cameras = None

        self.config = config

        self._data_train = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}
        self._data_valid = {"2d": np.zeros((0, 16, 2), dtype=np.float32), "3d": np.zeros((0, 16, 3), dtype=np.float32)}

        self.mean_2d = 0.0
        self.std_2d = 0.0
        self.mean_3d = 0.0
        self.std_3d = 0.0

        self.load_data(path, config)

    def scale_2d(self, data):

        assert data.shape[-1] == 2

        data_2d_re = data.reshape((-1, 2)).T
        data_2d_re = np.vstack((data_2d_re, np.ones((1, data_2d_re.shape[1]))))

        scale_matrix = np.eye(3)
        scale_matrix[0, 0] = (3.51/5.19)*(9./16.)
        scale_matrix[1, 1] = (3.51/5.19)*(9./16.)

        data_2d_re = np.matmul(scale_matrix, data_2d_re)
        data_2d_re = data_2d_re[:2, :].T.reshape((-1, 16, 2))

        return data_2d_re

    def load_data(self, path, config):
        filename = os.path.splitext(os.path.basename(path))[0]

        data = np.load(path, allow_pickle=True, encoding='latin1')['data'].item()
        data_train = data['train']
        data_valid = data['test']


        self.input_shape = config['input_shape']    
        self.model_info = config['model_info']

        indices_to_select, _ =  dataset_indices(self.config['dataset'], self.input_shape['num_joints'])

        # rects_train = data_train["combined_r2d"]
        # rects_valid = data_valid["combined_r2d"]

        self._data_train['2d'] = data_train["combined_2d"][:, indices_to_select,  :]
        self._data_train['3d'] = data_train["combined_3d_cam"][:, indices_to_select,  :]*1000

        self._data_valid['2d'] = data_valid["combined_2d"][:, indices_to_select,  :]
        self._data_valid['3d'] = data_valid["combined_3d_cam"][:, indices_to_select,  :]*1000

        if self.model_info['normalize_2d_to_minus_one_to_one']:
            self._data_valid['2d'] = normalize_screen_coordinates(self._data_valid['2d'], w=1080, h=1920)

        if self.model_info['output_3d'] == 'millimeters':
            self._data_valid['3d'] /= 1000.0  # convert from mm to meters 

        self._data_train['2d'] = self.root_center_vectorized(np.array(self._data_train['2d']))
        self._data_valid['2d'] = self.root_center_vectorized(np.array(self._data_valid['2d']), self.model_info['root_center_2d_test_input'])

        self._data_train['3d'] = self.root_center_vectorized(np.array(self._data_train['3d']))
        self._data_valid['3d'] = self.root_center_vectorized(np.array(self._data_valid['3d']))

        self.mean_3d = np.mean(self._data_train['3d'], axis=0)
        self.std_3d = np.std(self._data_train['3d'], axis=0)
        self.mean_2d = np.mean(self._data_train['2d'], axis=0)
        self.std_2d = np.std(self._data_train['2d'], axis=0)


        if self.model_info['trained_on_normalized_data']:
                self._data_valid['3d'] = normalize_data(self._data_valid['3d'], self.mean_3d, self.std_3d, skip_root=True)
                self._data_valid['2d'] = normalize_data(self._data_valid['2d'], self.mean_2d, self.std_2d, skip_root=True)

    def data(self):
        if self.model_info['flattened_coordinate_model']:
            self._data_valid['3d'] = self._data_valid['3d'].reshape((-1, self.input_shape['num_joints']*3))
            self._data_valid['2d'] = self._data_valid['2d'].reshape((-1, self.input_shape['num_joints']*2))
        return self._data_valid['3d'], self._data_valid['2d']
    
    def root_center_vectorized(self, data, apply_root_centering=True):
        if apply_root_centering:
            data = data - data[:, :1, :]
        return data


