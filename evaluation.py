import torch
import numpy as np
from common.loss import mpjpe, p_mpjpe 

from model import Model  # Import Model from model.py
from dataset import Dataset  # If Dataset is from dataset.py

from common.generators import Evaluate_Generator

class Evaluation:
    def __init__(self, model: Model, dataset: Dataset):
        self.model = model
        self.dataset = dataset

    def execute_evaluation(self, test_generator, dataset, action=None, return_predictions=False):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0

        with torch.no_grad():
            model_pos = self.model.get_model()
            model_pos.eval()
            N = 0
            
            # Test-time augmentation (if enabled)
            if self.model.config['test_time_augmentation']: 
                #print("Beginning actual evaluation...")
                #start_time = time.time()
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
                    predicted_3d_pos_flip[:, :, self.dataset.joints_left + self.dataset.joints_right] = predicted_3d_pos_flip[:, :,
                                                                            self.dataset.joints_right + self.dataset.joints_left]

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

                    epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)
                    # print("Finished actual evaluation...")

            else:
                #print("Beginning actual evaluation...")
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

                    epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)
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

    def evaluate(self):
        print("Evaluating...")
        all_actions, all_actions_by_subject = self.dataset.organize_actions()

        errors_p1 = []
        errors_p2 = []
        for action in all_actions.keys():
            poses_act, poses_2d_act = self.dataset.data(all_actions, action)
            #print(poses_act)
            #print(poses_2d_act)
            #print("Entering generator...")
            gen = Evaluate_Generator(self.model.config['batch_size']//self.model.config['stride'], None, poses_act, poses_2d_act, self.model.config['stride'],
                                            pad=self.model.pad, causal_shift=self.model.causal_shift, augment=self.model.config['test_time_augmentation'],
                                            shuffle=False,
                                            kps_left=self.dataset.kps_left, kps_right=self.dataset.kps_right, joints_left=self.dataset.joints_left,
                                            joints_right=self.dataset.joints_right)
            #print("Exiting generator...")

            #print("Beginning actual evaluation...")
            e1, e2 = self.execute_evaluation(gen, self.dataset, action)
            errors_p1.append(e1)
            errors_p2.append(e2)
            #print("Finished actual evaluation...")

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        

