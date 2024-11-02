import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import os

from common.loss import *
from common.generators import Evaluate_Generator

class Model:
    def __init__(self, config):
        self.config = config
        self.receptive_field = config['number_of_frames']
        self.pad = (self.receptive_field - 1) // 2
        self.causal_shift = 0
        self.model = None
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(self.device)
        # self.load_model()
    
    def load_model(self, config):
        # Load the model path from the config
        model_path = os.path.join(config['checkpoint'], config['evaluate'])
        print('Loading model from', model_path)
        self.model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.eval()
        model_params = sum(p.numel() for p in self.model.parameters())
        print('INFO: Trainable parameter count:', model_params/1e6, 'Million')
        
        # make model parallel
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
        

    def evaluate(self, test_generator, dataset, action=None, return_predictions=False):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0

        with torch.no_grad():
            model_pos = self.model
            model_pos.eval()
            N = 0

            # Test-time augmentation (if enabled)
            if self.config['test_time_augmentation']: 
                for _, batch, batch_2d, batch_2d_flip in test_generator.next_epoch():
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    inputs_2d_flip = torch.from_numpy(batch_2d_flip.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda()
                        inputs_2d_flip = inputs_2d_flip.cuda()

                    # Positional model
                    predicted_3d_pos = model_pos(inputs_2d)
                    predicted_3d_pos_flip = model_pos(inputs_2d_flip)
                    predicted_3d_pos_flip[:, :, :, 0] *= -1
                    predicted_3d_pos_flip[:, :, dataset.joints_left + dataset.joints_right] = predicted_3d_pos_flip[:, :,
                                                                            dataset.joints_right + dataset.joints_left]

                    predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                                keepdim=True)

                    if return_predictions:
                        return predicted_3d_pos.squeeze().cpu().numpy()

                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                    inputs_3d[:, :, 0] = 0

                    error = mpjpe(predicted_3d_pos, inputs_3d)

                    epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                    predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                    epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos,
                                                                                                    inputs)

            else:
                for _, batch, batch_2d in test_generator.next_epoch():
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda()

                    # Positional model
                    predicted_3d_pos = model_pos(inputs_2d)

                    if return_predictions:
                        return predicted_3d_pos.squeeze().cpu().numpy()

                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                    inputs_3d[:, :, 0] = 0

                    error = mpjpe(predicted_3d_pos, inputs_3d)

                    epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                    predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                    epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos,
                                                                                                inputs)

        if action is None:
            print('----------')
        else:
            print('----' + action + '----')
        e1 = (epoch_loss_3d_pos / N) * 1000
        e2 = (epoch_loss_3d_pos_procrustes / N) * 1000

        print('Test time augmentation:', test_generator.augment_enabled())
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('----------')
        return e1, e2

    def eval_h36m(self, dataset):
        print('Evaluating...')

        all_actions, all_actions_by_subject = dataset.organize_actions()
        
        def run_evaluation(actions, action_filter=None): 
            errors_p1 = []
            errors_p2 = []

            for action_key in actions.keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action_key.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_act, poses_2d_act = dataset.fetch_actions(actions[action_key], self.config)

                gen = Evaluate_Generator(self.config['batch_size']//self.config['stride'], None, poses_act, poses_2d_act, self.config['stride'],
                                        pad=self.pad, causal_shift=self.causal_shift, augment=self.config['test_time_augmentation'],
                                        shuffle=False,
                                        kps_left=dataset.kps_left, kps_right=dataset.kps_right, joints_left=dataset.joints_left,
                                        joints_right=dataset.joints_right)

                e1, e2 = self.evaluate(gen, dataset, action_key)
                errors_p1.append(e1)
                errors_p2.append(e2)

            print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
            print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')

        action_filter = None if self.config['actions'] == '*' else self.config['actions'].split(',')
        if not self.config['by_subject']: 
            run_evaluation(all_actions, action_filter)
        else:
            for subject in all_actions_by_subject.keys():
                print('Evaluating on subject', subject)
                run_evaluation(all_actions_by_subject[subject], action_filter)
                print('')