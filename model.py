import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import os

from common.loss import *
from common.generators import UnchunkedGenerator_Seq

class Model:
    def __init__(self, config):
        self.config = config
        self.receptive_field = config['number_of_frames']
        self.pad = (self.receptive_field - 1) // 2
        self.causal_shift = 0
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.load_model()
    
    def load_model(self, config):
        # Load the model path from the config
        model_path = os.path.join(config['checkpoint'], config['evaluate'])
        print('Loading model from', model_path)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        model_params = sum(p.numel() for p in self.model.parameters())
        print('INFO: Trainable parameter count:', model_params/1e6, 'Million')

        # Make model parallel
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
        
    
    def eval_data_prepare(self, inputs_2d, inputs_3d):
        receptive_field = self.receptive_field
        assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
        inputs_2d_p = torch.squeeze(inputs_2d)
        inputs_3d_p = torch.squeeze(inputs_3d)

        if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
            out_num = inputs_2d_p.shape[0] // receptive_field + 1
        elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
            out_num = inputs_2d_p.shape[0] // receptive_field

        eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
        eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

        for i in range(out_num-1):
            eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
            eval_input_3d[i,:,:,:] = inputs_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        if inputs_2d_p.shape[0] < receptive_field:
            from torch.nn import functional as F
            pad_right = receptive_field - inputs_2d_p.shape[0]
            inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
            inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
            inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
        if inputs_3d_p.shape[0] < receptive_field:
            pad_right = receptive_field - inputs_3d_p.shape[0]
            inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
            inputs_3d_p = F.pad(inputs_3d_p, (0,pad_right), mode='replicate')
            inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
        eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
        eval_input_3d[-1,:,:,:] = inputs_3d_p[-receptive_field:,:,:]

        return eval_input_2d.to(self.device), eval_input_3d.to(self.device)

    def evaluate(self, test_generator, dataset, action=None, return_predictions=False, use_trajectory_model=False, newmodel=None):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        N = 0

        with torch.no_grad():
            model_eval = self.model  # Use the model loaded in this class
            model_eval.eval()

            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_3d = torch.from_numpy(batch.astype('float32'))

                ##### apply test-time-augmentation (following Videopose3d)
                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip [:, :, :, 0] *= -1
                inputs_2d_flip[:, :, dataset.kps_left + dataset.kps_right,:] = inputs_2d_flip[:, :, dataset.kps_right + dataset.kps_left,:]

                ##### convert size
                inputs_3d_p = inputs_3d
                inputs_2d, inputs_3d = self.eval_data_prepare(inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = self.eval_data_prepare(inputs_2d_flip, inputs_3d_p)

                inputs_2d = inputs_2d.to(self.device)
                inputs_2d_flip = inputs_2d_flip.to(self.device)
                inputs_3d = inputs_3d.to(self.device)
                    
                inputs_3d[:, :, 0] = 0  # Remove global offset
                
                predicted_3d_pos = model_eval(inputs_2d)
                predicted_3d_pos_flip = model_eval(inputs_2d_flip)
                predicted_3d_pos_flip[:, :, :, 0] *= -1
                predicted_3d_pos_flip[:, :, dataset.joints_left + dataset.joints_right] = predicted_3d_pos_flip[:, :,
                                                                    dataset.joints_right + dataset.joints_left]
                predicted_3d_pos = (predicted_3d_pos + predicted_3d_pos_flip) / 2

                if return_predictions:
                    return predicted_3d_pos.squeeze().cpu().numpy()

                error = mpjpe(predicted_3d_pos, inputs_3d)

                epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()
                epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos_np = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos_np, inputs)

                # Compute velocity error
                epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos_np, inputs)

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

    def eval_h36m(self, dataset):
        print('Evaluating...')
        all_actions = {}
        all_actions_by_subject = {}
        action_filter = None if self.config['actions'] == '*' else self.config['actions'].split(',')

        for subject in dataset.subjects_test:
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
                poses_2d = dataset.keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])

                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

            stride = self.config['downsample']
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
                                        pad=self.pad, causal_shift=self.causal_shift, augment=self.config['test_time_augmentation'],
                                        kps_left=dataset.kps_left, kps_right=dataset.kps_right, joints_left=dataset.joints_left,
                                        joints_right=dataset.joints_right)
                e1, e2, e3, ev = self.evaluate(gen, dataset, action_key)

                errors_p1.append(e1)
                errors_p2.append(e2)
                errors_p3.append(e3)
                errors_vel.append(ev)

            print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
            print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
            print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
            print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')


        if not self.config['by_subject']:
            run_evaluation(all_actions, action_filter)
        else:
            for subject in all_actions_by_subject.keys():
                print('Evaluating on subject', subject)
                run_evaluation(all_actions_by_subject[subject], action_filter)
                print('')