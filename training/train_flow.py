# Copyright 2026 The SeededGrasp Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import dataclasses
import os
import sys
import time

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from hydra.utils import instantiate, to_absolute_path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omegaconf import DictConfig

import models.flow
from utils import math_utils
from utils.normalization import normalize_pc, normalize_q
from utils_data.augmentors import (
    sample_pc_noise_random,
    sample_seed_pt_noise_random,
    sample_xy_shifts_random,
    sample_z_rotations_random,
)
from utils_data.rgbd_scene_dataset import collate_fn_rgbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclasses.dataclass
class TrainState:
    seededgrasp_model: torch.nn.Module
    train_step: int
    eval_step: int
    writer: SummaryWriter
    optimizer: torch.optim.Optimizer
    epoch: int


def kaiming_init(model):
    """
    Initialize all Linear and Conv layers in the model using
    Kaiming (He) uniform initialization for ReLU activations.
    """
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def train(cfg: DictConfig, state: TrainState, dataloader: DataLoader):
    # Setup
    state.seededgrasp_model.train()
    state.optimizer.zero_grad()

    # Get underlying model (unwrap DataParallel if needed)
    model = (
        state.seededgrasp_model.module
        if isinstance(state.seededgrasp_model, nn.DataParallel)
        else state.seededgrasp_model
    )

    loss_history = []

    for i, data in enumerate(tqdm(dataloader, desc=f"Epoch: {state.epoch}")):
        state.train_step += 1

        # Setup data
        if "gnndatasetrgbdscene" in cfg.dataset.dataloader_name:
            (
                _,  # field_names
                q,
                robot_adj,
                robot_features,
                rest_pose,
                obj_adj_list,  # scene_adj as list of sparse tensors
                obj_features,  # scene_features used as obj_features
                robot_name,
                _,  # object_id
                _,  # scene_id
                seed_pt,  # Pre-selected first closest point (B, 1, 3)
            ) = data

            q = q.to(device, non_blocking=True)
            obj_features = obj_features.to(device, non_blocking=True)
            seed_pt = seed_pt.to(device, non_blocking=True)

            # Transfer sparse adj matrices to GPU, then convert to batched dense
            # Do conversion on GPU to avoid CPU bottleneck (GPU has spare compute)
            obj_adj = torch.stack(
                [adj.to(device, non_blocking=True).to_dense() for adj in obj_adj_list], dim=0
            )

            if cfg.dataset.augment_rot:
                grasp_trans = q[:, :3]
                grasp_rotation = math_utils.robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
                random_z_rotations = sample_z_rotations_random(q.shape[0], device=device)
                grasp_rotation = random_z_rotations @ grasp_rotation
                grasp_trans = random_z_rotations @ grasp_trans.unsqueeze(1).transpose(-1, -2)
                obj_features = (random_z_rotations @ obj_features.transpose(-1, -2)).transpose(
                    -1, -2
                )
                seed_pt = (random_z_rotations @ seed_pt.transpose(-1, -2)).transpose(
                    -1, -2
                )  # Apply same rotation to seed point
                q[:, :3] = grasp_trans.squeeze()
                q[:, 3:9] = grasp_rotation[:, :, :2].transpose(1, 2).reshape(q.shape[0], -1)
            if cfg.dataset.augment_trans:
                random_xy_shifts = sample_xy_shifts_random(q.shape[0], device=device)
                q[:, :3] = q[:, :3] + random_xy_shifts
                obj_features = obj_features + random_xy_shifts.unsqueeze(1)
                seed_pt = seed_pt + random_xy_shifts.unsqueeze(1)
            if cfg.dataset.augment_pc:
                random_pc_noise = sample_pc_noise_random(
                    q.shape[0], obj_features.shape[1], device=device
                )
                obj_features = obj_features + random_pc_noise
            if getattr(cfg.dataset, "augment_seed_pt", False):
                seed_pt = seed_pt + sample_seed_pt_noise_random(
                    q.shape[0], cfg.dataset.seed_pt_noise_std, device=device
                )

            mask = torch.isnan(q)
            q_zeroes = torch.nan_to_num(q, nan=0.0)

            if cfg.dataset.normalize_q:
                q_zeroes = normalize_q(cfg, q_zeroes, robot_name)
                obj_features = normalize_pc(cfg, obj_features)
                seed_pt = normalize_pc(cfg, seed_pt)
                q = q_zeroes.clone()
                q[mask] = np.nan
        else:
            raise Exception("Unknown dataloader")

        # Sample t and x
        # t ~ Beta(a, b) biased toward t -> 1 (following the Pi0 paper), so the
        # loss emphasises the late, high-precision part of the trajectory.
        t = 0.999 * torch.distributions.beta.Beta(cfg.model.beta_a, cfg.model.beta_b).sample(
            (obj_features.shape[0], 1)
        ).to(device)
        t_noise = torch.clamp(t + torch.randn_like(t) * 0.01, 0.0, 0.999)
        x0 = model.cond_prob_path.sample_x_init(q_zeroes)
        x0[mask] = 0.0
        x, x0 = model.cond_prob_path.sample_conditional_path(q_zeroes, t_noise, x0)
        x = x.to(device)
        x0 = x0.to(device)

        # Forward pass with dropout conditioning (handled inside model)
        drop_mask = None
        if cfg.model.zero_null_token:
            batch_size = obj_features.shape[0]
            drop_mask = torch.rand(batch_size, device=device) < cfg.train.drop_condition_prob

        # seed_pt: one candidate drawn per sample in __getitem__ (B, 1, 3)

        # Convert robot names to indices on CPU, then transfer to GPU (avoid sync)
        robot_name_idx = torch.tensor(
            [model.robot_name_to_idx[rn] for rn in robot_name], dtype=torch.long
        )
        robot_name_idx = robot_name_idx.to(device, non_blocking=True)

        outputs = state.seededgrasp_model(
            x, t, obj_features, obj_adj, robot_name_idx, seed_pt, drop_mask
        )

        # Compute loss
        loss, loss_breakdown = model.calc_loss(outputs, x, q, t, x0, e=state.epoch)

        # Move loss to CPU to prevent GPU memory accumulation
        loss_history.append(loss.item())

        # Compute training metrics
        if state.train_step % 2 == 0:
            # Move to CPU explicitly to avoid implicit sync during TensorBoard write
            state.writer.add_scalar("train/all_loss", loss.item(), state.train_step)

            if loss_breakdown is not None:
                state.writer.add_scalar(
                    "train/trans_loss", loss_breakdown[0].item(), state.train_step
                )
                state.writer.add_scalar(
                    "train/rot_loss", loss_breakdown[1].item(), state.train_step
                )
                state.writer.add_scalar(
                    "train/joint_loss", loss_breakdown[2].item(), state.train_step
                )

        # Backward pass
        loss.backward()
        state.optimizer.step()
        state.optimizer.zero_grad()

    # Compute epoch loss from scalar values (already on CPU)
    epoch_loss = sum(loss_history) / len(loss_history)
    print(f"Epoch {state.epoch}, train loss: {epoch_loss}")
    return epoch_loss


def validate(cfg, state: TrainState, dataloader: DataLoader):
    # Get underlying model (unwrap DataParallel if needed)
    model = (
        state.seededgrasp_model.module
        if isinstance(state.seededgrasp_model, nn.DataParallel)
        else state.seededgrasp_model
    )

    with torch.no_grad():
        state.seededgrasp_model.eval()
        loss_history = []

        for _, data in enumerate(tqdm(dataloader, desc=f"Validation Epoch: {state.epoch}")):
            state.eval_step += 1

            # Setup data
            if "gnndatasetrgbdscene" in cfg.dataset.dataloader_name:
                (
                    _,  # field_names
                    q,
                    robot_adj,
                    robot_features,
                    rest_pose,
                    obj_adj_list,  # scene_adj as list of sparse tensors
                    obj_features,  # scene_features used as obj_features
                    robot_name,
                    _,  # object_id
                    _,  # scene_id
                    seed_pt,  # Pre-selected first closest point (B, 1, 3)
                ) = data
                q = q.to(device, non_blocking=True)
                obj_features = obj_features.to(device, non_blocking=True)
                seed_pt = seed_pt.to(device, non_blocking=True)

                # Transfer sparse adj matrices to GPU, then convert to batched dense on GPU
                obj_adj = torch.stack(
                    [adj.to(device, non_blocking=True).to_dense() for adj in obj_adj_list], dim=0
                )

                if cfg.dataset.augment_rot:
                    grasp_trans = q[:, :3]
                    grasp_rotation = math_utils.robust_compute_rotation_matrix_from_ortho6d(
                        q[:, 3:9]
                    )
                    random_z_rotations = sample_z_rotations_random(q.shape[0], device=device)
                    grasp_rotation = random_z_rotations @ grasp_rotation
                    grasp_trans = random_z_rotations @ grasp_trans.unsqueeze(1).transpose(-1, -2)
                    obj_features = (random_z_rotations @ obj_features.transpose(-1, -2)).transpose(
                        -1, -2
                    )
                    seed_pt = (random_z_rotations @ seed_pt.transpose(-1, -2)).transpose(
                        -1, -2
                    )  # Apply same rotation to seed point
                    q[:, :3] = grasp_trans.squeeze()
                    q[:, 3:9] = grasp_rotation[:, :, :2].transpose(1, 2).reshape(q.shape[0], -1)
                if cfg.dataset.augment_trans:
                    random_xy_shifts = sample_xy_shifts_random(q.shape[0], device=device)
                    q[:, :3] = q[:, :3] + random_xy_shifts
                    obj_features = obj_features + random_xy_shifts.unsqueeze(1)
                    seed_pt = seed_pt + random_xy_shifts.unsqueeze(1)
                if cfg.dataset.augment_pc:
                    random_pc_noise = sample_pc_noise_random(
                        q.shape[0], obj_features.shape[1], device=device
                    )
                    obj_features = obj_features + random_pc_noise
                if getattr(cfg.dataset, "augment_seed_pt", False):
                    seed_pt = seed_pt + sample_seed_pt_noise_random(
                        q.shape[0], cfg.dataset.seed_pt_noise_std, device=device
                    )

                mask = torch.isnan(q)
                q_zeroes = torch.nan_to_num(q, nan=0.0)

                if cfg.dataset.normalize_q:
                    q_zeroes = normalize_q(cfg, q_zeroes, robot_name)
                    obj_features = normalize_pc(cfg, obj_features)
                    seed_pt = normalize_pc(cfg, seed_pt)
                    q = q_zeroes.clone()
                    q[mask] = np.nan
            else:
                raise Exception("Unknown dataloader")

            # Sample t and x
            t = torch.rand(obj_features.shape[0], 1).to(device)
            x0 = model.cond_prob_path.sample_x_init(q_zeroes)
            x0[mask] = 0.0
            x, x0 = model.cond_prob_path.sample_conditional_path(q_zeroes, t, x0)
            x = x.to(device)
            x0 = x0.to(device)

            # seed_pt: one candidate drawn per sample in __getitem__ (B, 1, 3)

            # Convert robot names to indices on CPU, then transfer to GPU (avoid sync)
            robot_name_idx = torch.tensor(
                [model.robot_name_to_idx[rn] for rn in robot_name], dtype=torch.long
            )
            robot_name_idx = robot_name_idx.to(device, non_blocking=True)

            outputs = state.seededgrasp_model(x, t, obj_features, obj_adj, robot_name_idx, seed_pt)

            # Compute loss
            loss, loss_breakdown = model.calc_loss(outputs, x, q, t, x0, e=state.epoch)

            # Move loss to CPU to prevent GPU memory accumulation
            loss_history.append(loss.item())

            # Compute validation metrics
            if state.eval_step % 2 == 0:
                # Move to CPU explicitly to avoid implicit sync
                state.writer.add_scalar("val/all_loss", loss.item(), state.eval_step)

                if loss_breakdown is not None:
                    state.writer.add_scalar(
                        "val/trans_loss", loss_breakdown[0].item(), state.eval_step
                    )
                    state.writer.add_scalar(
                        "val/rot_loss", loss_breakdown[1].item(), state.eval_step
                    )
                    state.writer.add_scalar(
                        "val/joint_loss", loss_breakdown[2].item(), state.eval_step
                    )

        # Compute epoch loss from scalar values (already on CPU)
        epoch_loss = sum(loss_history) / len(loss_history)
        print(f"Epoch {state.epoch}, validation loss: {epoch_loss}")
        return epoch_loss


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    # General setup
    start_time = time.time()
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    # Setup logging
    log_basedir = to_absolute_path(cfg.train.log_basedir)
    log_dir = os.path.join(log_basedir, f"{cfg.train.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    tb_dir = os.path.join(log_dir, cfg.train.tb_dir)
    weight_dir = os.path.join(log_dir, cfg.train.weights_dir)

    if not os.path.exists(log_basedir):
        os.makedirs(log_basedir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    writer = SummaryWriter(log_dir=tb_dir)

    # Setup dataloaders
    robot_name_list = []
    for name in cfg.dataset.robot_name_list:
        robot_name_list.append(cfg.dataset.robot_name_mapping[name])

    dataset_basedir = to_absolute_path(cfg.dataset.dataset_basedir)
    gripper_alignment_rot6d = getattr(cfg.dataset, "gripper_alignment_rot6d", None)
    canonical_rot6d = getattr(cfg.dataset, "canonical_rot6d", None)
    train_dataset_kwargs = dict(
        dataset_basedir=dataset_basedir,
        mode="train",
        robot_name_list=robot_name_list,
        pad_q=cfg.dataset.pad_q,
    )
    val_dataset_kwargs = dict(
        dataset_basedir=dataset_basedir,
        mode="validate",
        robot_name_list=robot_name_list,
        pad_q=cfg.dataset.pad_q,
        # Deterministic seed point (the nearest candidate) so val is comparable
        # across epochs; training draws a random candidate each time.
        random_seed_pt=False,
    )
    if "gnndatasetrgbdscene" in cfg.dataset.dataloader_name:
        train_dataset_kwargs["gripper_alignment_rot6d"] = gripper_alignment_rot6d
        train_dataset_kwargs["canonical_rot6d"] = canonical_rot6d
        val_dataset_kwargs["gripper_alignment_rot6d"] = gripper_alignment_rot6d
        val_dataset_kwargs["canonical_rot6d"] = canonical_rot6d

    if cfg.dataset.dataloader_name == "gnndatasetrgbdscenehierarchical":
        train_dataset_kwargs["train_scene_percent"] = getattr(
            cfg.dataset, "train_scene_percent", 100.0
        )
        train_dataset_kwargs["scene_subset_seed"] = getattr(cfg.train, "seed", None)

    train_dataset = instantiate(cfg.dataset.dataloader, **train_dataset_kwargs)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_rgbd,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = instantiate(cfg.dataset.dataloader, **val_dataset_kwargs)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_rgbd,
        pin_memory=True,
        persistent_workers=True,
    )

    robot_models_path = os.path.join(
        to_absolute_path(cfg.dataset.dataset_basedir), "hand_models.pt"
    )
    robot_models = torch.load(robot_models_path, weights_only=False, map_location=device)
    not_present_robots = []
    for k in robot_models.keys():
        if k not in robot_name_list:
            not_present_robots.append(k)
    for k in not_present_robots:
        del robot_models[k]

    # Initialize model
    seededgrasp_model = models.flow.SeededGraspFlow(
        cfg=cfg, robot_models=robot_models, robot_models_path=robot_models_path
    ).to(device)

    # Wrap in DataParallel if multiple GPUs are available
    # NOTE: DataParallel has unbalanced memory usage (GPU 0 uses more memory)
    # For better memory balance, consider switching to DistributedDataParallel (DDP)
    # DDP requires torchrun: torchrun --nproc_per_node=N train_flow.py
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        print(
            "WARNING: DataParallel may cause memory imbalance. GPU 0 typically uses 30-50% more memory."
        )
        seededgrasp_model = nn.DataParallel(seededgrasp_model)

    total_params = 0
    model_for_params = (
        seededgrasp_model.module
        if isinstance(seededgrasp_model, nn.DataParallel)
        else seededgrasp_model
    )
    for name, parameter in model_for_params.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
        print(f"{name}: {params} params")
    print(f"Total Trainable Params: {total_params}")

    # Initialize weights
    print("Initializing weights with Kaiming initialization")
    model_to_init = (
        seededgrasp_model.module
        if isinstance(seededgrasp_model, nn.DataParallel)
        else seededgrasp_model
    )
    kaiming_init(model_to_init)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        seededgrasp_model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.99),
    )

    # Initialize training state
    train_step = 0
    eval_step = 0

    train_state = TrainState(
        seededgrasp_model=seededgrasp_model,
        writer=writer,
        optimizer=optimizer,
        train_step=train_step,
        eval_step=eval_step,
        epoch=0,
    )

    # Training loop
    min_val_epoch_loss = np.inf
    # Get underlying model for saving (unwrap DataParallel if needed)
    model_to_save = (
        seededgrasp_model.module
        if isinstance(seededgrasp_model, nn.DataParallel)
        else seededgrasp_model
    )

    for e in range(cfg.train.epochs):
        train(cfg, train_state, train_dataloader)
        if e % 4 == 0:
            val_epoch_loss = validate(cfg, train_state, val_dataloader)
        train_state.epoch += 1

        # Save model weights
        if e % 4 == 0:
            torch.save(model_to_save.state_dict(), os.path.join(weight_dir, cfg.train.weights_name))
            if min_val_epoch_loss > val_epoch_loss:
                torch.save(model_to_save.state_dict(), os.path.join(weight_dir, "best_val.pth"))
                min_val_epoch_loss = val_epoch_loss

        # Decay learning rate
        try:
            if e % cfg.train.lr_decay_epochs == 0 and e != 0:
                for param_group in train_state.optimizer.param_groups:
                    param_group["lr"] *= 1 - cfg.train.lr_decay
                print(f"Learning rate decayed to {train_state.optimizer.param_groups[0]['lr']}")
        except Exception:
            pass

    torch.save(model_to_save.state_dict(), os.path.join(weight_dir, cfg.train.weights_name))

    writer.close()
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
