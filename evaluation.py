import torch
import numpy as np
from common.loss import mpjpe, p_mpjpe, p_mpjpe_per_joint, mpjpe_per_joint
from model import Model  # Import Model from model.py
from dataset import Dataset  # If Dataset is from dataset.py
from torch.utils.data import DataLoader
from common.generators import PoseGenerator
from common.utils import unnormalize_data
import os
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from poseutils.metrics import calculate_jpe


class Evaluation:
    def __init__(self, model: Model, dataset: Dataset):
        self.model = model
        self.dataset = dataset

    def _print_input_target_shapes(self, inputs_2d, targets_3d):
        """
        Helper to print input shapes (one-time).
        """
        print("==> Inputs 2D Shape: ", inputs_2d.shape)
        print("==> Target 3D Shape: ", targets_3d.shape)

    def _print_output_shapes(self, outputs_3d, targets_3d):
        """
        Helper to print output shapes (one-time).
        """
        print("==> Outputs 3D Shape: ", outputs_3d.shape)
        print("==> Target 3D Shape: ", targets_3d.shape)

    def _print_one_data_point(self, inputs_2d, outputs_3d, targets_3d):
        """
        Helper to print a single data point (input, target, output) for debugging.
        """
        def to_numpy(t):
            return t.cpu().numpy() if isinstance(t, torch.Tensor) else t

        inputs_2d_np = to_numpy(inputs_2d)
        outputs_3d_np = to_numpy(outputs_3d)
        targets_3d_np = to_numpy(targets_3d)

        print("==> Input Data Point:\n", inputs_2d_np[0])
        print("---------------")
        print("==> Target Data Point:\n", targets_3d_np[0])
        print("---------------")
        print("==> Output Data Point:\n", outputs_3d_np[0])


    def _compute_batch_errors(
        self,
        outputs_3d,
        targets_3d,
        epoch_loss_3d_pos,
        epoch_loss_3d_pos_procrustes,
        N
    ):
        """
        Helper that computes MPJPE and P-MPJPE for the current batch, accumulates 
        them, and returns updated metrics as well as rolling averages.
        """
        # MPJPE error
        error = mpjpe(outputs_3d, targets_3d)
        epoch_loss_3d_pos += (targets_3d.shape[0] * targets_3d.shape[1] * error.item())

        # N is total number of 3D points processed so far
        N += targets_3d.shape[0] * targets_3d.shape[1]

        # P-MPJPE error
        inputs = targets_3d.cpu().numpy().reshape(-1, targets_3d.shape[-2], targets_3d.shape[-1])
        predicted_3d_pos = outputs_3d.cpu().numpy().reshape(-1, outputs_3d.shape[-2], outputs_3d.shape[-1])
        epoch_loss_3d_pos_procrustes += (targets_3d.shape[0] * targets_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs))

        # Rolling average so far
        avg_mpjpe = epoch_loss_3d_pos / (N + 1e-8)
        avg_pmpjpe = epoch_loss_3d_pos_procrustes / (N + 1e-8)

        return epoch_loss_3d_pos, epoch_loss_3d_pos_procrustes, N, avg_mpjpe, avg_pmpjpe

    def execute_evaluation(self, data_loader, model_pos, device):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0

        # initialise the collectors if we will need them 
        if self.model.config.get('save_predictions') or self.model.config.get('per_joint_error'):
            all_outputs, all_targets, all_inputs = [], [], []


        # These booleans control one-time printing
        printInput_Target_Shape = True
        printOutputAndTargetShape = True
        printOneTargetAndOutputDataPoint = True

        # Get your PyTorch model
        model_pos = self.model.get_model()
        if self.model.config.get("model_type") in ["Pytorch", "JIT"]:
            model_pos.eval()

        N = 0  # count of total points processed so far

        progress = Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TaskProgressColumn(),
            "• Elapsed:", TimeElapsedColumn(),
            "• ETA:", TimeRemainingColumn(),
            "• MPJPE:", TextColumn("{task.fields[mpjpe]:.2f} mm"),
            "• P-MPJPE:", TextColumn("{task.fields[pmjpe]:.2f} mm"),
        )

        with progress:
            task = progress.add_task("Evaluating...", total=len(data_loader), mpjpe=0.0, pmjpe=0.0)

            with torch.no_grad():
                for batch_idx, (targets_3d, inputs_2d) in enumerate(data_loader):
                    
                    # -- Print input and targetshapes (only once) --
                    if printInput_Target_Shape:
                        self._print_input_target_shapes(inputs_2d, targets_3d)
                        printInput_Target_Shape = False

                    # -- Inference --
                    if self.model.config.get("model_type") in ["Pytorch", "JIT"]:
                        inputs_2d = inputs_2d.to(device)
                        # If some remove_hypothesis logic is used
                        if self.dataset.model_info.get('remove_hypothesis', False):
                            targets_3d = targets_3d.to(device)
                            outputs_3d = model_pos(inputs_2d, targets_3d)
                            outputs_3d = outputs_3d[0].squeeze(2)
                        else:
                            outputs_3d = model_pos(inputs_2d).cpu()
                    else:
                        inputs_np = inputs_2d.cpu().numpy().astype(np.float32)
                        outputs_np = self.model.predict(inputs_np)
                        outputs_3d = torch.from_numpy(outputs_np)

                    if self.dataset.model_info['trained_on_normalized_data']:
                        targets_3d = targets_3d.reshape((-1, self.dataset.input_shape['num_joints'], 3))
                        outputs_3d = outputs_3d.reshape((-1, self.dataset.input_shape['num_joints'], 3))
                        inputs_2d = inputs_2d.reshape((-1, self.dataset.input_shape['num_joints'], 2)).cpu().numpy()
                        if self.dataset.config.get('test_using_train_mean_and_std', False):
                            targets_3d = unnormalize_data(targets_3d, self.dataset.train_mean_3d, self.dataset.train_std_3d, skip_root=True)
                            outputs_3d = unnormalize_data(outputs_3d, self.dataset.train_mean_3d, self.dataset.train_std_3d, skip_root=True)
                            inputs_2d = unnormalize_data(inputs_2d, self.dataset.train_mean_2d, self.dataset.train_std_2d, skip_root=True)
                        else:
                            targets_3d = unnormalize_data(targets_3d, self.dataset.mean_3d, self.dataset.std_3d, skip_root=True)
                            outputs_3d = unnormalize_data(outputs_3d, self.dataset.mean_3d, self.dataset.std_3d, skip_root=True)
                            inputs_2d = unnormalize_data(inputs_2d, self.dataset.mean_2d, self.dataset.std_2d, skip_root=True)

                    # Convert outputs and targets to torch if they are numpy
                    if isinstance(outputs_3d, np.ndarray):
                        outputs_3d = torch.from_numpy(outputs_3d).to(device)
                    if isinstance(targets_3d, np.ndarray):
                        targets_3d = torch.from_numpy(targets_3d).to(device)

                    # -- Print output shapes (only once) --
                    if printOutputAndTargetShape:
                        self._print_output_shapes(outputs_3d, targets_3d)
                        printOutputAndTargetShape = False

                    # -- Print one data point example (only once) --
                    if printOneTargetAndOutputDataPoint and self.dataset.config['print_sample']:
                        self._print_one_data_point(inputs_2d, outputs_3d, targets_3d)
                        printOneTargetAndOutputDataPoint = False
                    
                    # ─── inside the loop, where you append results ───────────────────────────────
                    if self.model.config.get('save_predictions') or self.model.config.get('per_joint_error'):
                        all_outputs.append(outputs_3d.cpu().numpy())
                        all_targets.append(targets_3d.cpu().numpy())
                        all_inputs.append(inputs_2d.cpu().numpy() if isinstance(inputs_2d, torch.Tensor) else inputs_2d)



                    # -- Compute errors (MPJPE, P-MPJPE) --
                    (epoch_loss_3d_pos, epoch_loss_3d_pos_procrustes, N, avg_mpjpe, avg_pmpjpe) = self._compute_batch_errors(outputs_3d, targets_3d, epoch_loss_3d_pos, epoch_loss_3d_pos_procrustes, N)

                    # -- Convert results to millimeters if needed --
                    if self.dataset.model_info['output_3d'] == 'meters':
                        e1 = avg_mpjpe
                        e2 = avg_pmpjpe
                    else:
                        e1 = avg_mpjpe * 1000
                        e2 = avg_pmpjpe * 1000

                    # Update the progress bar
                    progress.update(task, advance=1, mpjpe=e1, pmjpe=e2)


                # Only concatenate once if either flag is set
                if self.model.config['save_predictions'] or self.model.config['per_joint_error']:
                    all_outputs = np.concatenate(all_outputs, axis=0)
                    all_targets = np.concatenate(all_targets, axis=0)
                    all_inputs = np.concatenate(all_inputs, axis=0)

                if self.model.config['save_predictions']:
                    # Format folder name as Predictions_{DATASET}
                    dataset_folder = f"Predictions_{self.dataset.config['dataset'].upper()}"
                    os.makedirs(dataset_folder, exist_ok=True)

                    # Get model name without extension
                    model_filename = os.path.splitext(os.path.basename(self.model.config['evaluate']))[0]

                    # Save path: Predictions_{DATASET}/{model_filename}.npz
                    save_path = os.path.join(dataset_folder, f"{model_filename}.npz")

                    # Save predictions
                    np.savez_compressed(save_path, inputs2D=all_inputs, targets3D=all_targets, outputs3D=all_outputs)
                    print(f"Saved predictions to {save_path} with {all_outputs.shape[0]} predictions, "
                        f"{all_targets.shape[0]} targets, and {all_inputs.shape[0]} inputs")

                if self.model.config['per_joint_error']:
                    # Squeeze frame dimension if present
                    all_outputs_flat = all_outputs.squeeze(1) if all_outputs.ndim == 4 else all_outputs
                    all_targets_flat = all_targets.squeeze(1) if all_targets.ndim == 4 else all_targets

                    mpjpe_joints = mpjpe_per_joint(all_outputs_flat, all_targets_flat)
                    pmpjpe_joints = p_mpjpe_per_joint(all_outputs_flat, all_targets_flat)

                    # Convert to mm if needed
                    if self.dataset.model_info['output_3d'] != 'meters':
                        mpjpe_joints *= 1000
                        pmpjpe_joints *= 1000

                    # Print horizontal lists
                    mpjpe_list_str = " ".join([f"{val:.2f}" for val in mpjpe_joints])
                    pmpjpe_list_str = " ".join([f"{val:.2f}" for val in pmpjpe_joints])

                    print("Per-Joint MPJPE (mm):")
                    print(mpjpe_list_str)
                    print("\nPer-Joint P-MPJPE (mm):")
                    print(pmpjpe_list_str)

                    print(f"\nMean MPJPE: {mpjpe_joints.mean():.2f} mm")
                    print(f"Mean P-MPJPE: {pmpjpe_joints.mean():.2f} mm")



    def p_mpjpe_per_joint(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Per-joint PA-MPJPE (Protocol-2).
        predicted, target: (N, J, 3) numpy arrays
        returns          : (J,)     numpy array – mean error for each joint
        """
        # --- rigid alignment (same math as your p_mpjpe, but vectorised) -----------
        muX, muY = target.mean(1, keepdims=True), predicted.mean(1, keepdims=True)
        X0,  Y0  = target - muX, predicted - muY
        normX, normY = np.linalg.norm(X0, axis=(1, 2), keepdims=True), np.linalg.norm(Y0, axis=(1, 2), keepdims=True)
        X0,  Y0  = X0 / normX, Y0 / normY

        H   = np.matmul(X0.transpose(0, 2, 1), Y0)
        U,  _, Vt = np.linalg.svd(H)
        V   = Vt.transpose(0, 2, 1)
        R   = np.matmul(V, U.transpose(0, 2, 1))
        detR = np.sign(np.linalg.det(R))[:, None, None]
        R  *= detR
        tr  = (R * H).sum((1, 2), keepdims=True)

        a   = tr * normX / normY
        t   = muX - a * (muY @ R)
        pred_aligned = a * (predicted @ R) + t
        # ---------------------------------------------------------------------------

        # L2 error per joint, then average over N frames
        return np.linalg.norm(pred_aligned - target, axis=2).mean(axis=0)



    def evaluate(self):
        print('\n==> Evaluating...')
        
        if self.model.config['dataset'] in {'h36m','gpa', 'surreal', '3dpw'}:
            poses_3d_all, poses_2d_all = self.dataset.data()

            print("==> Raw Data Info")
            print("==> Full 3D Data Shape:", poses_3d_all.shape)
            print("==> Full 2D Data Shape:", poses_2d_all.shape)

        print("==> Video Model:", self.model.model_info['video_model'])
        valid_loader = DataLoader(
            PoseGenerator(poses_3d_all, poses_2d_all, video_mode=self.model.model_info['video_model']),
            batch_size=self.model.config['input_shape'].get('batch_size'), 
            shuffle=False,
            num_workers=self.model.config['num_workers'], 
            pin_memory=True,
            drop_last=True
        )

        # Move model to GPU or CPU depending on config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.execute_evaluation(valid_loader, self.model, device)
        exit(0)
