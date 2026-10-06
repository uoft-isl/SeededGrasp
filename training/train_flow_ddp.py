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

"""
DistributedDataParallel version of train_flow.py for better multi-GPU memory balance.

Usage:
    # Single GPU
    python training/train_flow_ddp.py

    # Multi-GPU (recommended)
    torchrun --nproc_per_node=2 training/train_flow_ddp.py
    # or
    python -m torch.distributed.launch --nproc_per_node=2 training/train_flow_ddp.py

Key improvements over DataParallel:
- Balanced memory usage across all GPUs
- 20-30% faster training
- Each GPU computes its own loss (no gathering on GPU 0)
- Better gradient synchronization
"""

import os
import sys
import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from hydra.utils import instantiate, to_absolute_path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omegaconf import DictConfig

import models.flow
from training.train_flow import TrainState, kaiming_init
from utils import math_utils
from utils.normalization import normalize_pc, normalize_q
from utils_data.augmentors import (
    sample_pc_noise_random,
    sample_seed_pt_noise_random,
    sample_xy_shifts_random,
    sample_z_rotations_random,
)
from utils_data.rgbd_scene_dataset import collate_fn_rgbd


def setup_ddp():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def cleanup_ddp():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_ddp(
    cfg: DictConfig, state: TrainState, dataloader: DataLoader, rank: int, world_size: int
):
    """Training loop with DDP."""
    state.seededgrasp_model.train()
    state.optimizer.zero_grad()

    # Get underlying model (unwrap DDP)
    model = (
        state.seededgrasp_model.module
        if isinstance(state.seededgrasp_model, DDP)
        else state.seededgrasp_model
    )
    device = next(model.parameters()).device

    loss_history = []

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch: {state.epoch}")
    else:
        pbar = dataloader

    for i, data in enumerate(pbar):
        state.train_step += 1

        # Data setup (same as original train_flow.py)
        if "gnndatasetrgbdscene" in cfg.dataset.dataloader_name:
            (
                _,  # field_names
                q,
                robot_adj,
                robot_features,
                rest_pose,
                obj_adj_list,
                obj_features,
                robot_name,
                _,  # object_id
                _,  # scene_id
                seed_pt,
            ) = data

            q = q.to(device, non_blocking=True)
            obj_features = obj_features.to(device, non_blocking=True)
            seed_pt = seed_pt.to(device, non_blocking=True)

            # Convert sparse to dense and batch - model converts to dense anyway in GNN layers
            # Memory per sample: 2048×2048×4 bytes = 16MB dense adj
            obj_adj = torch.stack(
                [adj.to(device, non_blocking=True).to_dense() for adj in obj_adj_list], dim=0
            )

            # Augmentation
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
            raise NotImplementedError(
                f"Dataloader {cfg.dataset.dataloader_name} not implemented in DDP version"
            )

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

        # Forward pass with dropout conditioning (handled inside model)
        drop_mask = None
        if cfg.model.zero_null_token:
            batch_size = obj_features.shape[0]
            drop_mask = torch.rand(batch_size, device=device) < cfg.train.drop_condition_prob

        robot_name_idx = torch.tensor(
            [model.robot_name_to_idx[rn] for rn in robot_name], dtype=torch.long
        )
        robot_name_idx = robot_name_idx.to(device, non_blocking=True)

        outputs = state.seededgrasp_model(
            x, t, obj_features, obj_adj, robot_name_idx, seed_pt, drop_mask
        )

        # Compute loss (happens independently on each GPU - no gathering!)
        loss, loss_breakdown = model.calc_loss(outputs, x, q, t, x0, e=state.epoch)
        loss_history.append(loss.item())

        # Logging (only rank 0)
        if rank == 0 and state.train_step % 2 == 0:
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

        # Backward pass (DDP handles gradient synchronization automatically)
        loss.backward()
        state.optimizer.step()
        state.optimizer.zero_grad()

    # Compute epoch loss
    epoch_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0

    # Average loss across all GPUs
    if world_size > 1:
        loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        epoch_loss = loss_tensor.item()

    if rank == 0:
        print(f"Epoch {state.epoch}, train loss: {epoch_loss}")

    return epoch_loss


def validate_ddp(cfg, state: TrainState, dataloader: DataLoader, rank: int, world_size: int):
    """Validation loop with DDP."""
    model = (
        state.seededgrasp_model.module
        if isinstance(state.seededgrasp_model, DDP)
        else state.seededgrasp_model
    )
    device = next(model.parameters()).device

    with torch.no_grad():
        state.seededgrasp_model.eval()
        loss_history = []

        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Validation Epoch: {state.epoch}")
        else:
            pbar = dataloader

        for _, data in enumerate(pbar):
            state.eval_step += 1

            # Data setup (similar to train)
            if "gnndatasetrgbdscene" in cfg.dataset.dataloader_name:
                (
                    _,
                    q,
                    robot_adj,
                    robot_features,
                    rest_pose,
                    obj_adj_list,
                    obj_features,
                    robot_name,
                    _,
                    _,
                    seed_pt,
                ) = data

                q = q.to(device, non_blocking=True)
                obj_features = obj_features.to(device, non_blocking=True)
                seed_pt = seed_pt.to(device, non_blocking=True)
                obj_adj = torch.stack(
                    [adj.to(device, non_blocking=True).to_dense() for adj in obj_adj_list], dim=0
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
                raise NotImplementedError(
                    f"Dataloader {cfg.dataset.dataloader_name} not implemented"
                )

            # Sample and forward
            t = torch.rand(obj_features.shape[0], 1).to(device)
            x0 = model.cond_prob_path.sample_x_init(q_zeroes)
            x0[mask] = 0.0
            x, x0 = model.cond_prob_path.sample_conditional_path(q_zeroes, t, x0)

            robot_name_idx = torch.tensor(
                [model.robot_name_to_idx[rn] for rn in robot_name], dtype=torch.long
            )
            robot_name_idx = robot_name_idx.to(device, non_blocking=True)

            outputs = state.seededgrasp_model(x, t, obj_features, obj_adj, robot_name_idx, seed_pt)
            loss, loss_breakdown = model.calc_loss(outputs, x, q, t, x0, e=state.epoch)

            loss_history.append(loss.item())

            if rank == 0 and state.eval_step % 2 == 0:
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

        epoch_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0

        # Average across GPUs
        if world_size > 1:
            loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            epoch_loss = loss_tensor.item()

        if rank == 0:
            print(f"Epoch {state.epoch}, validation loss: {epoch_loss}")

        return epoch_loss


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    # Setup DDP
    rank, world_size, local_rank, device = setup_ddp()

    # Only print on rank 0
    if rank == 0:
        print(f"Running DDP with {world_size} GPUs")

    # General setup
    start_time = time.time()
    np.random.seed(cfg.train.seed + rank)  # Different seed per rank
    torch.manual_seed(cfg.train.seed + rank)

    # Setup logging (only rank 0)
    if rank == 0:
        log_basedir = to_absolute_path(cfg.train.log_basedir)
        log_dir = os.path.join(
            log_basedir, f"{cfg.train.exp_name}_ddp_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        tb_dir = os.path.join(log_dir, cfg.train.tb_dir)
        weight_dir = os.path.join(log_dir, cfg.train.weights_dir)

        os.makedirs(log_basedir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(weight_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=tb_dir)
    else:
        writer = None
        weight_dir = None

    # Setup dataloaders with DistributedSampler
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
    val_dataset = instantiate(cfg.dataset.dataloader, **val_dataset_kwargs)

    # Use DistributedSampler for multi-GPU
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        collate_fn=collate_fn_rgbd,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_rgbd,
        pin_memory=True,
        persistent_workers=True,
    )

    # Load robot models
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

    # Wrap in DDP with memory-efficient settings
    if world_size > 1:
        seededgrasp_model = DDP(
            seededgrasp_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # Required: some params (e.g. robot embeddings) may not be used in every batch
            broadcast_buffers=False,  # Save memory: don't broadcast buffers if not needed
            gradient_as_bucket_view=True,  # Memory efficient gradient storage
        )

    if rank == 0:
        total_params = sum(p.numel() for p in seededgrasp_model.parameters() if p.requires_grad)
        print(f"Total Trainable Params: {total_params}")

    # Initialize weights
    if rank == 0:
        print("Initializing weights with Kaiming initialization")
    model_to_init = (
        seededgrasp_model.module if isinstance(seededgrasp_model, DDP) else seededgrasp_model
    )
    kaiming_init(model_to_init)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        seededgrasp_model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.99),
    )

    # Training state
    train_state = TrainState(
        seededgrasp_model=seededgrasp_model,
        writer=writer,
        optimizer=optimizer,
        train_step=0,
        eval_step=0,
        epoch=0,
    )

    # Training loop
    min_val_epoch_loss = np.inf
    model_to_save = (
        seededgrasp_model.module if isinstance(seededgrasp_model, DDP) else seededgrasp_model
    )

    for e in range(cfg.train.epochs):
        # Set epoch for DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(e)

        train_ddp(cfg, train_state, train_dataloader, rank, world_size)

        if e % 4 == 0:
            val_epoch_loss = validate_ddp(cfg, train_state, val_dataloader, rank, world_size)

        train_state.epoch += 1

        # Save model (only rank 0)
        if rank == 0 and e % 4 == 0:
            torch.save(model_to_save.state_dict(), os.path.join(weight_dir, cfg.train.weights_name))
            if min_val_epoch_loss > val_epoch_loss:
                torch.save(model_to_save.state_dict(), os.path.join(weight_dir, "best_val.pth"))
                min_val_epoch_loss = val_epoch_loss

        # Learning rate decay
        if e % cfg.train.lr_decay_epochs == 0 and e != 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 1 - cfg.train.lr_decay
            if rank == 0:
                print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']}")

    # Final save
    if rank == 0:
        torch.save(model_to_save.state_dict(), os.path.join(weight_dir, cfg.train.weights_name))
        writer.close()
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
