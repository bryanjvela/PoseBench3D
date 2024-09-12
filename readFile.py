import ipdb
import numpy as np

#from common.arguments import parse_args
import argparse

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math
import yaml

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections
from common.model_ktpformer import *


from common.loss import *
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from time import time
from common.utils import *
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    # Use argparse only for the config file path
    parser = argparse.ArgumentParser(description="Pass YAML config file path")
    parser.add_argument('-cfg', '--config', default='config.yaml', type=str, help='path to the config file')
    args = parser.parse_args()
    return args

def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Parse the command-line argument for the config file
    args = parse_args()

    # Load the config from the specified YAML file
    config = load_config(args.config)

    sys.path.append(config['model_location'])  # Add the directory, not the full path to the file

    if config['evaluate'] != '':
        description = "Evaluate!"
    elif config['evaluate'] == '':
        description = "Train!"

    # initial setting
    TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
    # tensorboard
    if not config['nolog']:
        writer = SummaryWriter(config['log']+'_'+TIMESTAMP)
        writer.add_text('description', description)
        writer.add_text('command', 'python ' + ' '.join(sys.argv))
        # logging setting
        logfile = os.path.join(config['log']+'_'+TIMESTAMP, 'training_logging.log')
        sys.stdout = Logger(logfile)
    print(description)
    print('python ' + ' '.join(sys.argv))
    print("CUDA Device Count: ", torch.cuda.device_count())
    
    # Sort the config dictionary by keys and print in alphabetical order
    def print_config_namespace_style(config):
        config_items = [f"{key}={repr(value)}" for key, value in sorted(config.items())]
        print("Namespace(" + ", ".join(config_items) + ")")
    print_config_namespace_style(config)

    # dataset loading
    print('Loading dataset...')
    dataset_path = 'data/data_3d_' + config['dataset'] + '.npz'
    if config['dataset'] == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    else:
        raise KeyError('Invalid dataset')
    

    # prepare data for use 
    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    print('Loading 2D detections...')
    keypoints = np.load('data/data_2d_' + config['dataset'] + '_' + config['keypoints'] + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    ###################
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_train = config['subjects_train'].split(',')
    subjects_semi = [] if not config['subjects_unlabeled'] else config['subjects_unlabeled'].split(',')
    if not config['render']:
        subjects_test = config['subjects_test'].split(',')
    else:
        subjects_test = [config['viz_subject']]

    
    def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []
        for subject in subjects:
            for action in keypoints[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): # Iterate across cameras
                        out_poses_3d.append(poses_3d[i])

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = config['downsample']
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]


        return out_camera_params, out_poses_3d, out_poses_2d

    action_filter = None if config['actions'] == '*' else config['actions'].split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

    # set receptive_field as number assigned
    receptive_field = config['number_of_frames'] 
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    if not config['nolog']: 
        writer.add_text(config['log']+'_'+TIMESTAMP + '/Receptive field', str(receptive_field))
    pad = (receptive_field -1) // 2 # Padding on each side
    min_loss = config['min_loss']
    width = cam['res_w']
    height = cam['res_h']
    num_joints = keypoints_metadata['num_joints']

    # Load the model path from the config
    model_path = os.path.join(config['checkpoint'], config['evaluate'])
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()

    causal_shift = 0
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params/1000000, 'Million')
    if not config['nolog']:
        writer.add_text(config['log']+'_'+TIMESTAMP + '/Trainable parameter count', str(model_params/1000000) + ' Million')


    # make model parallel
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()

    def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
        assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
        inputs_2d_p = torch.squeeze(inputs_2d)
        inputs_3d_p = torch.squeeze(inputs_3d)

        if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
            out_num = inputs_2d_p.shape[0] // receptive_field+1
        elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
            out_num = inputs_2d_p.shape[0] // receptive_field

        eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
        eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

        for i in range(out_num-1):
            eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
            eval_input_3d[i,:,:,:] = inputs_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        if inputs_2d_p.shape[0] < receptive_field:
            from torch.nn import functional as F
            pad_right = receptive_field-inputs_2d_p.shape[0]
            inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
            inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
            # inputs_2d_p = np.pad(inputs_2d_p, ((0, receptive_field-inputs_2d_p.shape[0]), (0, 0), (0, 0)), 'edge')
            inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
        if inputs_3d_p.shape[0] < receptive_field:
            pad_right = receptive_field-inputs_3d_p.shape[0]
            inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
            inputs_3d_p = F.pad(inputs_3d_p, (0,pad_right), mode='replicate')
            inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
        eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
        eval_input_3d[-1,:,:,:] = inputs_3d_p[-receptive_field:,:,:]

        return eval_input_2d, eval_input_3d
    

    def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False, newmodel=None):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        with torch.no_grad():
            if newmodel is not None:
                print('Loading comparison model')
                model_eval = newmodel
                chk_file_path = './checkpoint/best_epoch_2dgt.bin'
                print('Loading evaluate checkpoint of comparison model', chk_file_path)
                checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
                model_eval.load_state_dict(checkpoint['model_pos'], strict=False)
                model_eval.eval()
            else:
                model_eval = model  # Use the full model directly
                if not use_trajectory_model:
                    # load best checkpoint
                    if config['evaluate'] == '':
                        chk_file_path = os.path.join(config['checkpoint'], 'best_epoch.bin')
                        print('Loading best checkpoint', chk_file_path)
                    elif config['evaluate'] != '':
                        chk_file_path = os.path.join(config['checkpoint'], config['evaluate'])
                        print('Loading evaluate checkpoint', chk_file_path)

                    # Load the checkpoint, which contains model components
                    checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
                    model_eval = checkpoint
                    model_eval.cuda()
                    
                    model_eval.eval()
            # else:
                # model_traj.eval()
            N = 0
            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_3d = torch.from_numpy(batch.astype('float32'))

                ##### apply test-time-augmentation (following Videopose3d)
                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip [:, :, :, 0] *= -1
                inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]
                
                ##### convert size
                inputs_3d_p = inputs_3d
                if newmodel is not None:
                    def eval_data_prepare_pf(receptive_field, inputs_2d, inputs_3d):
                        inputs_2d_p = torch.squeeze(inputs_2d)
                        inputs_3d_p = inputs_3d.permute(1,0,2,3)
                        padding = int(receptive_field//2)
                        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
                        inputs_2d_p = F.pad(inputs_2d_p, (padding,padding), mode='replicate')
                        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
                        out_num = inputs_2d_p.shape[0] - receptive_field + 1
                        eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
                        for i in range(out_num):
                            eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
                        return eval_input_2d, inputs_3d_p
                    
                    inputs_2d, inputs_3d = eval_data_prepare_pf(81, inputs_2d, inputs_3d_p)
                    inputs_2d_flip, _ = eval_data_prepare_pf(81, inputs_2d_flip, inputs_3d_p)
                else:
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                    inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)


                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_flip = inputs_2d_flip.cuda()
                    inputs_3d = inputs_3d.cuda()
                    
                inputs_3d[:, :, 0] = 0
                
                predicted_3d_pos = model_eval(inputs_2d)
                predicted_3d_pos_flip = model_eval(inputs_2d_flip)
                predicted_3d_pos_flip[:, :, :, 0] *= -1
                predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                        joints_right + joints_left]
                for i in range(predicted_3d_pos.shape[0]):
                    predicted_3d_pos[i,:,:,:] = (predicted_3d_pos[i,:,:,:] + predicted_3d_pos_flip[i,:,:,:])/2
                

                if return_predictions:
                    return predicted_3d_pos.squeeze().cpu().numpy()

                error = mpjpe(predicted_3d_pos, inputs_3d)

            

                epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

                epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

                # Compute velocity error
                epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

        if action is None:
            print('----------')
        else:
            print('----'+action+'----')
        e1 = (epoch_loss_3d_pos / N)*1000
        e2 = (epoch_loss_3d_pos_procrustes / N)*1000
        e3 = (epoch_loss_3d_pos_scale / N)*1000
        ev = (epoch_loss_3d_vel / N)*1000
        print('Test time augmentation:', test_generator.augment_enabled())
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
        print('Velocity Error (MPJVE):', ev, 'mm')
        print('----------')

        return e1, e2, e3, ev
    

    def eval_h36m(model, dataset):
        print('Evaluating...')
        all_actions = {}
        all_actions_by_subject = {}
        for subject in subjects_test:
            if subject not in all_actions_by_subject:
                all_actions_by_subject[subject] = {}

            for action in dataset[subject].keys():
                action_name = action.split(' ')[0]
                if action_name not in all_actions:
                    all_actions[action_name] = []
                if action_name not in all_actions_by_subject[subject]:
                    all_actions_by_subject[subject][action_name] = []
                all_actions[action_name].append((subject, action))
                all_actions_by_subject[subject][action_name].append((subject, action))

        def fetch_actions(actions):
            out_poses_3d = []
            out_poses_2d = []

            for subject, action in actions:
                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])

                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

            stride = config['downsample']
            if stride > 1:
                # Downsample as requested
                for i in range(len(out_poses_2d)):
                    out_poses_2d[i] = out_poses_2d[i][::stride]
                    if out_poses_3d is not None:
                        out_poses_3d[i] = out_poses_3d[i][::stride]

            return out_poses_3d, out_poses_2d

        def run_evaluation(actions, action_filter=None):
            errors_p1 = []
            errors_p2 = []
            errors_p3 = []
            errors_vel = []
            # joints_errs_list=[]

            for action_key in actions.keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action_key.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_act, poses_2d_act = fetch_actions(actions[action_key])
                gen = UnchunkedGenerator_Seq(None, poses_act, poses_2d_act,
                                        pad=pad, causal_shift=causal_shift, augment=config['test_time_augmentation'],
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                        joints_right=joints_right)
                e1, e2, e3, ev = evaluate(gen, action_key)
                
                errors_p1.append(e1)
                errors_p2.append(e2)
                errors_p3.append(e3)
                errors_vel.append(ev)

            print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
            print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
            print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
            print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')


        if not config['by_subject']:
            run_evaluation(all_actions, action_filter)
        else:
            for subject in all_actions_by_subject.keys():
                print('Evaluating on subject', subject)
                run_evaluation(all_actions_by_subject[subject], action_filter)
                print('')

    
    eval_h36m(model, dataset)

if __name__ == "__main__":
     main()
