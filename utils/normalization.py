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

"""Min-max normalization of grasp poses and point clouds to [-1, 1].

Translations use the symmetric workspace bound `dataset.trans_limits`; joint
angles use per-gripper bounds from `dataset.dof_limits`, so grippers with
different joint limits, scales and units share one normalized range.

This lives outside `training/` so that `models.flow` can use it without
importing a training entrypoint (which imports `models.flow` in turn).
"""

import torch


def normalize_pc(cfg, pc):
    """Normalize point cloud xyz coordinates to [-1, 1]."""
    pc = pc.clone()
    trans_limits = cfg.dataset.trans_limits
    for i in range(3):  # xyz
        pc[:, :, i] = 2 * ((pc[:, :, i] - (-trans_limits)) / (2 * trans_limits)) - 1

    return pc


def normalize_q(cfg, q, robot_name):
    # Normalize different components of q separately
    # Split q into its components
    q_trans = q[:, :3]  # Translation (x, y, z)
    q_rot = q[:, 3:9]  # Rotation (6D representation)
    q_joints = q[:, 9:]  # Joint angles

    # Normalize translation (assuming symmetric limits)
    trans_limits = cfg.dataset.trans_limits
    q_trans_normalized = 2 * ((q_trans - (-trans_limits)) / (2 * trans_limits)) - 1

    # Normalize joints - handle different robots in batch
    q_joints_normalized = torch.zeros_like(q_joints)

    # Get unique robot names in this batch
    unique_robots = (
        torch.unique(robot_name) if torch.is_tensor(robot_name) else list(set(robot_name))
    )

    for robot in unique_robots:
        # Find indices for this robot type
        if torch.is_tensor(robot_name):
            robot_mask = robot_name == robot
        else:
            robot_mask = torch.tensor([r == robot for r in robot_name], device=q.device)

        if robot_mask.any():
            robot_str = robot.item() if torch.is_tensor(robot) else robot
            joint_min = torch.tensor(cfg.dataset.dof_limits[robot_str][0], device=q.device)
            joint_max = torch.tensor(cfg.dataset.dof_limits[robot_str][1], device=q.device)

            # Normalize joints for this robot type
            robot_joints = q_joints[robot_mask]
            robot_joints_norm = 2 * ((robot_joints - joint_min) / (joint_max - joint_min)) - 1
            q_joints_normalized[robot_mask] = robot_joints_norm

    # Reconstruct normalized q
    q = torch.cat([q_trans_normalized, q_rot, q_joints_normalized], dim=1)
    return q


def unnormalize_q(cfg, q, robot_name):
    # Split q into its components
    q_trans = q[:, :3]  # Translation (x, y, z)
    q_rot = q[:, 3:9]  # Rotation (6D representation)
    q_joints = q[:, 9:]  # Joint angles

    # Unnormalize translation (assuming symmetric limits)
    trans_limits = cfg.dataset.trans_limits
    q_trans_unnormalized = ((q_trans + 1) / 2.0) * (2 * trans_limits) - trans_limits

    # Unnormalize joints - handle different robots in batch
    q_joints_unnormalized = torch.zeros_like(q_joints)

    # Get unique robot names in this batch
    unique_robots = (
        torch.unique(robot_name) if torch.is_tensor(robot_name) else list(set(robot_name))
    )

    for robot in unique_robots:
        # Find indices for this robot type
        if torch.is_tensor(robot_name):
            robot_mask = robot_name == robot
        else:
            robot_mask = torch.tensor([r == robot for r in robot_name], device=q.device)

        if robot_mask.any():
            robot_str = robot.item() if torch.is_tensor(robot) else robot
            joint_min = torch.tensor(cfg.dataset.dof_limits[robot_str][0], device=q.device)
            joint_max = torch.tensor(cfg.dataset.dof_limits[robot_str][1], device=q.device)

            robot_joints = q_joints[robot_mask]
            robot_joints_unnorm = ((robot_joints + 1) / 2.0) * (joint_max - joint_min) + joint_min
            q_joints_unnormalized[robot_mask] = robot_joints_unnorm

    # Reconstruct unnormalized q
    q = torch.cat([q_trans_unnormalized, q_rot, q_joints_unnormalized], dim=1)
    return q
