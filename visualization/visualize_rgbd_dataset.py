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
Visualization script for RGBD scene dataset.

Visualizes scene point clouds with robot grasp poses overlaid.
"""

import os
import sys

import hydra
import plotly.graph_objects as go
import plotly.io as pio
import torch
from hydra.utils import instantiate, to_absolute_path
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import math_utils
from utils_data.augmentors import (
    sample_pc_noise_random,
    sample_xy_shifts_random,
    sample_z_rotations_random,
)
from utils_data.rgbd_scene_dataset import collate_fn_rgbd

pio.renderers.default = "browser"


def visualize_sample(
    q: torch.Tensor,
    scene_features: torch.Tensor,
    hand_model,
    robot_name: str,
    object_name: str,
    scene_name: str,
    closest_object_points: torch.Tensor = None,
    apply_augmentation: bool = False,
    show_robot_mesh: bool = False,
):
    """
    Visualize a single grasp sample.

    Args:
        q: Grasp configuration (position + rot6d + dofs)
        scene_features: Scene point cloud features
        hand_model: Hand model for visualization
        robot_name: Name of the robot/gripper
        object_name: Name of the target object
        scene_name: Name of the scene
        apply_augmentation: Whether to apply data augmentation for visualization
        show_robot_mesh: Whether to show robot mesh (slower) or just point cloud
    """
    fig = go.Figure()

    # Apply augmentation if requested
    if apply_augmentation:
        # Random Z rotation
        grasp_trans = q[:, :3]
        grasp_rotation = math_utils.robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
        random_z_rotations = sample_z_rotations_random(q.shape[0], device=grasp_rotation.device)
        grasp_rotation = random_z_rotations @ grasp_rotation
        grasp_trans = random_z_rotations @ grasp_trans.unsqueeze(1).transpose(-1, -2)
        scene_features = (random_z_rotations @ scene_features.transpose(-1, -2)).transpose(-1, -2)
        q[:, :3] = grasp_trans.squeeze()
        q[:, 3:9] = grasp_rotation[:, :, :2].transpose(1, 2).reshape(q.shape[0], -1)

        # Random XY shift
        random_xy_shifts = sample_xy_shifts_random(q.shape[0], device=grasp_rotation.device)
        q[:, :3] = q[:, :3] + random_xy_shifts
        scene_features = scene_features + random_xy_shifts.unsqueeze(1)

        # Random point cloud noise
        random_pc_noise = sample_pc_noise_random(
            q.shape[0], scene_features.shape[1], device=grasp_rotation.device
        )
        scene_features = scene_features + random_pc_noise

    # Take first sample if batched
    q = q[0].unsqueeze(0)
    scene_features = scene_features[0].unsqueeze(0)

    # Visualize robot
    if show_robot_mesh:
        # Show full mesh (slower but more detailed)
        vis_data = hand_model.get_plotly_data(q=q.to(torch.device("cuda")))
        for d in vis_data:
            fig.add_trace(d)
    else:
        # Show point cloud (faster)
        # Determine how many DOFs to use based on robot
        num_pose_params = 9  # 3 translation + 6 rotation
        total_q_len = q.shape[1]
        num_dofs = total_q_len - num_pose_params

        # Get robot surface points at grasp pose
        robot_q = q[:, : num_pose_params + num_dofs]

        # Handle NaN padding
        robot_q_clean = torch.nan_to_num(robot_q, nan=0.0)

        robot_features = hand_model.get_surface_points(
            q=robot_q_clean.to(torch.device("cuda")), downsample=True
        )

        fig.add_trace(
            go.Scatter3d(
                x=robot_features.squeeze()[:, 0].cpu(),
                y=robot_features.squeeze()[:, 1].cpu(),
                z=robot_features.squeeze()[:, 2].cpu(),
                mode="markers",
                marker=dict(size=2, color="red"),
                name=f"Robot: {robot_name}",
            )
        )

    # Visualize scene point cloud
    fig.add_trace(
        go.Scatter3d(
            x=scene_features.squeeze()[:, 0].cpu(),
            y=scene_features.squeeze()[:, 1].cpu(),
            z=scene_features.squeeze()[:, 2].cpu(),
            mode="markers",
            marker=dict(size=2, color="blue"),
            name="Scene Point Cloud",
        )
    )

    # Add grasp position marker
    grasp_pos = q[0, :3].cpu()
    fig.add_trace(
        go.Scatter3d(
            x=[grasp_pos[0].item()],
            y=[grasp_pos[1].item()],
            z=[grasp_pos[2].item()],
            mode="markers",
            marker=dict(size=8, color="green", symbol="diamond"),
            name="Grasp Position",
        )
    )

    # Visualize closest object points if available
    if closest_object_points is not None and not torch.isnan(closest_object_points).all():
        # Apply same augmentation to closest points if needed
        augmented_pts = closest_object_points.clone()
        if apply_augmentation:
            # This is already augmented in the same way as scene_features above
            pass

        fig.add_trace(
            go.Scatter3d(
                x=augmented_pts[:, 0].cpu(),
                y=augmented_pts[:, 1].cpu(),
                z=augmented_pts[:, 2].cpu(),
                mode="markers",
                marker=dict(size=6, color="orange", symbol="circle", opacity=1.0),
                name="Closest Object Points (8)",
            )
        )

    # Layout
    fig.update_layout(
        title=f"Scene: {scene_name} | Object: {object_name} | Gripper: {robot_name}",
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode="cube",
        ),
        width=1000,
        height=800,
        showlegend=True,
    )

    return fig


@hydra.main(config_path="../configs", config_name="config_rgbd_scene", version_base=None)
def main(cfg):
    """Main visualization loop."""

    # Load hand models
    hand_models_path = os.path.join(to_absolute_path(cfg.dataset.dataset_basedir), "hand_models.pt")
    if os.path.exists(hand_models_path):
        hand_models = torch.load(hand_models_path, weights_only=False)
    else:
        print(f"Warning: hand_models.pt not found at {hand_models_path}")
        print("Robot visualization will be limited.")
        hand_models = {}

    # Build robot name list
    robot_name_list = []
    for name in cfg.dataset.robot_name_list:
        mapped_name = cfg.dataset.robot_name_mapping.get(name, name)
        robot_name_list.append(mapped_name)

    # Create dataloader
    dataset = instantiate(
        cfg.dataset.dataloader,
        dataset_basedir=to_absolute_path(cfg.dataset.dataset_basedir),
        mode="validate",
        robot_name_list=robot_name_list,
        pad_q=cfg.dataset.pad_q,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_rgbd,
    )

    print("\nDataset Statistics:")
    print(f"  Available grippers: {dataset.available_grippers}")
    print(f"  Total scenes: {dataset.total_scenes}")
    print(f"  Total objects: {dataset.total_objects}")
    print(f"  Total grasps: {dataset.total_grasps}")
    print("\nShowing samples... (close each plot to see the next)")

    # Visualization settings
    num_samples_to_show = 30
    apply_augmentation = False  # Set to True to see augmented data
    show_robot_mesh = False  # Set to True for detailed mesh (slower)

    dataloader_iter = iter(dataloader)

    for i in range(num_samples_to_show):
        try:
            sample = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            sample = next(dataloader_iter)

        # Unpack sample
        sample_data = sample

        # Handle different sample formats
        if len(sample_data) == 11:
            # New format with closest_object_points
            (
                field_names,
                q,
                robot_adj,
                robot_features,
                rest_pose,
                scene_adj,
                scene_features,
                gripper_id,
                object_id,
                scene_id,
                closest_object_points,
            ) = sample_data
        else:
            # Old format without closest_object_points
            (
                field_names,
                q,
                robot_adj,
                robot_features,
                rest_pose,
                scene_adj,
                scene_features,
                gripper_id,
                object_id,
                scene_id,
            ) = sample_data
            closest_object_points = None

        # Get robot name (it's a list when batched)
        robot_name = gripper_id[0] if isinstance(gripper_id, list) else gripper_id
        object_name = object_id[0] if isinstance(object_id, list) else object_id
        scene_name = scene_id[0] if isinstance(scene_id, list) else scene_id

        print(f"\nSample {i+1}/{num_samples_to_show}:")
        print(f"  Gripper: {robot_name}")
        print(f"  Scene: {scene_name}")
        print(f"  Object: {object_name}")
        print(f"  Q shape: {q.shape}")
        print(f"  Scene features shape: {scene_features.shape}")

        # Get hand model for this gripper
        if robot_name in hand_models:
            hand_model = hand_models[robot_name]
        else:
            print(f"  Warning: No hand model for {robot_name}, skipping robot visualization")
            hand_model = None

        if hand_model is not None:
            mask = ~torch.isnan(q)

            # Extract closest points if available
            closest_pts = None
            if closest_object_points is not None:
                # closest_object_points is batched, take first sample
                closest_pts = (
                    closest_object_points[0]
                    if len(closest_object_points.shape) == 3
                    else closest_object_points
                )

            fig = visualize_sample(
                q=q.clone()[mask].unsqueeze(0),
                scene_features=scene_features.clone(),
                hand_model=hand_model,
                robot_name=robot_name,
                object_name=object_name,
                scene_name=scene_name,
                closest_object_points=closest_pts,
                apply_augmentation=apply_augmentation,
                show_robot_mesh=show_robot_mesh,
            )
            fig.show()
        else:
            # Just show scene if no hand model
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=scene_features.squeeze()[:, 0],
                    y=scene_features.squeeze()[:, 1],
                    z=scene_features.squeeze()[:, 2],
                    mode="markers",
                    marker=dict(size=2, color="blue"),
                    name="Scene Point Cloud",
                )
            )
            fig.update_layout(
                title=f"Scene: {scene_name} | Object: {object_name} (no hand model)",
                scene=dict(
                    xaxis=dict(range=[-1, 1]),
                    yaxis=dict(range=[-1, 1]),
                    zaxis=dict(range=[-1, 1]),
                    aspectmode="cube",
                ),
                width=1000,
                height=800,
            )
            fig.show()


if __name__ == "__main__":
    main()
