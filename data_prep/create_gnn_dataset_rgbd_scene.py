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
Data preprocessing script for RGBD scene dataset.

This script processes multi-camera RGBD data into scene point clouds and
organizes grasp data for efficient training.

Dataset structure expected:
    scene_train/
        gripper_1/
            scene_a.json  (contains grasps for objects in scene_a)
            scene_b.json
            ...
        gripper_2/
            ...
    scene_val/
        gripper_1/
            ...

Each scene directory should contain:
    scene_name/
        scene_name.json  (object poses)
        camera_north_depth.pcd
        camera_south_depth.pcd
        camera_east_depth.pcd
        camera_west_depth.pcd
        camera_bev_depth.pcd
"""

import json
import os
import sys

import hydra
import numpy as np
import open3d as o3d
import torch
import trimesh as tm
from hydra.utils import to_absolute_path
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.general_utils import get_handmodel
from utils.gnn_utils import generate_adj_mat_feats
from utils.math_utils import get_rot6d_from_quat

device = "cpu"


def load_object_mesh(object_id: str, object_dir: str) -> tm.Trimesh:
    """
    Load object mesh from GSO or YCB dataset.
    """
    gso_obj_names = set()
    gso_file = os.path.join(object_dir, "gso_object_ids.txt")
    if os.path.exists(gso_file):
        with open(gso_file, "r") as f:
            for line in f:
                gso_obj_names.add(line.strip())

    if object_id in gso_obj_names:
        obj_mesh_path = os.path.join(object_dir, f"GSO/{object_id}/meshes/model.obj")
    else:
        obj_mesh_path = os.path.join(object_dir, f"YCB/{object_id}/textured.obj")

    if not os.path.exists(obj_mesh_path):
        return None

    return tm.load(obj_mesh_path)


# Number of nearest object points stored per grasp as seed-point candidates.
SEED_PT_CANDIDATES = 8


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) to select representative points from a point cloud.

    Args:
        points: Input point cloud of shape (N, 3)
        num_samples: Number of points to sample

    Returns:
        Sampled points of shape (num_samples, 3)
    """
    if num_samples >= len(points):
        # If we need more points than available, pad with duplicates
        if len(points) == 0:
            raise ValueError("Cannot sample from empty point cloud")
        indices = np.arange(len(points))
        extra_indices = np.random.choice(len(points), num_samples - len(points), replace=True)
        indices = np.concatenate([indices, extra_indices])
        return points[indices]

    sampled_indices = np.zeros(num_samples, dtype=int)
    distances = np.full(len(points), np.inf)

    # Start from a random point
    sampled_indices[0] = np.random.randint(len(points))

    for i in range(1, num_samples):
        sampled_point = points[sampled_indices[i - 1]]
        diff = points - sampled_point
        diff[points[:, 2] < 0.005] *= 0.3
        dists = np.linalg.norm(diff, axis=1)
        distances = np.minimum(distances, dists)
        sampled_indices[i] = np.argmax(distances)

    return points[sampled_indices]


def clean_and_merge_point_clouds(
    pcd_paths: list,
    min_xy: float = -0.4,
    max_xy: float = 0.4,
    voxel_size: float = 0.005,
) -> np.ndarray:
    """
    Load, merge, filter, and downsample point clouds from multiple cameras.

    Args:
        pcd_paths: List of paths to .pcd files from different cameras
        min_xy: Lower bound for X and Y spatial filtering
        max_xy: Upper bound for X and Y spatial filtering
        voxel_size: Voxel size for downsampling to merge close points

    Returns:
        Cleaned and merged point cloud as numpy array of shape (N, 3)
    """
    all_points = []

    for pcd_path in pcd_paths:
        if os.path.exists(pcd_path):
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            if len(points) > 0:
                all_points.append(points)

    if not all_points:
        raise ValueError(f"No valid point clouds found in {pcd_paths}")

    # Merge all point clouds
    combined_points = np.concatenate(all_points, axis=0)

    # Spatial filtering (XY plane boundaries)
    mask = (
        (combined_points[:, 0] >= min_xy)
        & (combined_points[:, 0] <= max_xy)
        & (combined_points[:, 1] >= min_xy)
        & (combined_points[:, 1] <= max_xy)
    )
    filtered_points = combined_points[mask]

    if len(filtered_points) == 0:
        raise ValueError("No points remain after spatial filtering")

    # Voxel downsampling to merge close points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return np.asarray(downsampled_pcd.points)


def process_scene_point_cloud(
    scene_dir: str,
    cameras: list,
    num_scene_pts: int,
    min_xy: float,
    max_xy: float,
    voxel_size: float,
) -> np.ndarray:
    """
    Process a single scene's multi-camera RGBD data into a unified point cloud.

    Args:
        scene_dir: Path to the scene directory
        cameras: List of camera names (e.g., ['camera_north_depth', ...])
        num_scene_pts: Target number of points in the output
        min_xy: Spatial filtering lower bound
        max_xy: Spatial filtering upper bound
        voxel_size: Voxel size for merging close points

    Returns:
        Point cloud of shape (num_scene_pts, 3)
    """
    pcd_paths = [os.path.join(scene_dir, f"{cam}.pcd") for cam in cameras]

    # Clean and merge multi-camera point clouds
    merged_pc = clean_and_merge_point_clouds(
        pcd_paths, min_xy=min_xy, max_xy=max_xy, voxel_size=voxel_size
    )

    # Use FPS to get exactly num_scene_pts points
    scene_pc = farthest_point_sampling(merged_pc, num_scene_pts)

    return scene_pc


@hydra.main(config_path="../configs", config_name="config_rgbd_scene", version_base=None)
def main(cfg):
    """Main preprocessing pipeline."""

    print("=" * 50)
    print("RGBD Scene Dataset Preprocessing")
    print("=" * 50)

    # =========================================================================
    # 1. Create robot point cloud data
    # =========================================================================
    print("\n[1/3] Creating robot point cloud data...")

    hand_model = {}
    data_dict = {}

    for robot_name in tqdm(cfg.dataset.robot_name_list, desc="Processing robots"):
        # Handle robot name mapping if needed
        mapped_name = cfg.dataset.robot_name_mapping.get(robot_name, robot_name)

        hand_model[mapped_name] = get_handmodel(
            mapped_name,
            1,
            torch.device("cuda"),
            1.0,
            data_dir=to_absolute_path(cfg.dataset.dataset_basedir),
        )

        # Compute rest pose
        joint_lower = np.array(hand_model[mapped_name].revolute_joints_q_lower.cpu().reshape(-1))
        joint_upper = np.array(hand_model[mapped_name].revolute_joints_q_upper.cpu().reshape(-1))
        joint_mid = (joint_lower + joint_upper) / 2
        joints_q = (joint_mid + joint_lower) / 2

        rest_pose = (
            torch.from_numpy(np.concatenate([np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]), joints_q]))
            .unsqueeze(0)
            .to(device)
            .float()
        )

        surface_points = (
            hand_model[mapped_name]
            .get_surface_points(rest_pose.cuda(), downsample=True)
            .cpu()
            .squeeze(0)
        )

        robot_adj, robot_features = generate_adj_mat_feats(
            surface_points.numpy(), knn=cfg.dataset.knn
        )

        data_dict[mapped_name] = (
            robot_adj,
            robot_features,
            rest_pose,
            mapped_name,
        )

    robot_data_path = os.path.join(
        to_absolute_path(cfg.dataset.dataset_basedir), "gnn_robot_adj_point_clouds_new.pt"
    )
    torch.save(data_dict, robot_data_path)
    print(f"Robot data saved to {robot_data_path}")

    # =========================================================================
    # 2. Process scene point clouds
    # =========================================================================
    print("\n[2/3] Processing scene point clouds...")

    scene_data = {}

    # Collect all unique scene directories from train and val
    all_scene_dirs = set()
    for split_dir in [cfg.dataset.scene_train_dir, cfg.dataset.scene_val_dir]:
        split_path = to_absolute_path(split_dir)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist, skipping...")
            continue

        # Each gripper subdirectory contains JSON files referencing scenes
        for gripper_name in os.listdir(split_path):
            gripper_dir = os.path.join(split_path, gripper_name)
            if not os.path.isdir(gripper_dir):
                continue

            for json_file in os.listdir(gripper_dir):
                if not json_file.endswith(".json"):
                    continue

                try:
                    with open(os.path.join(gripper_dir, json_file), "r") as f:
                        grasp_data = json.load(f)
                except Exception as e:
                    print(os.path.join(gripper_dir, json_file))
                    raise e

                # Determine scene directory from grasp data
                scene_id = grasp_data.get("scene_id", json_file.replace(".json", ""))
                # Assume scene directories are stored somewhere - adjust path as needed
                scene_dir = os.path.join(
                    to_absolute_path(cfg.dataset.dataset_basedir), "scenes", scene_id
                )
                if os.path.exists(scene_dir):
                    all_scene_dirs.add((scene_id, scene_dir))

    # Process each unique scene
    for scene_id, scene_dir in tqdm(all_scene_dirs, desc="Processing scenes"):
        try:
            scene_pc = process_scene_point_cloud(
                scene_dir=scene_dir,
                cameras=list(cfg.dataset.cameras),
                num_scene_pts=cfg.dataset.num_scene_pts,
                min_xy=cfg.dataset.spatial_bounds.min_xy,
                max_xy=cfg.dataset.spatial_bounds.max_xy,
                voxel_size=cfg.dataset.voxel_size,
            )

            scene_adj, scene_features = generate_adj_mat_feats(scene_pc, knn=cfg.dataset.knn)

            scene_data[scene_id] = {
                "scene_adj": scene_adj,
                "scene_features": scene_features,
            }
        except Exception as e:
            print(f"Warning: Failed to process scene {scene_id}: {e}")
            continue

    scene_data_path = os.path.join(
        to_absolute_path(cfg.dataset.dataset_basedir), "gnn_scene_adj_point_clouds_new.pt"
    )
    torch.save(scene_data, scene_data_path)
    print(f"Scene data saved to {scene_data_path}")

    # =========================================================================
    # 3. Process grasp data with hierarchical indexing
    # =========================================================================
    print("\n[3/3] Processing grasp data...")

    def process_grasps_for_split(split_dir: str, split_name: str) -> dict:
        """Process grasps for a single split (train or val)."""
        split_path = to_absolute_path(split_dir)

        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist")
            return {}

        # Hierarchical structure: gripper -> scene -> object -> grasps
        grasp_index = {}

        # Cache for object meshes and scene pose data
        object_mesh_cache = {}
        scene_pose_cache = {}
        object_points_cache = {}  # Cache for sampled object points in local frame

        object_dir = os.path.join(to_absolute_path(cfg.dataset.dataset_basedir), "objects")

        for gripper_name in tqdm(os.listdir(split_path), desc=f"Processing {split_name}"):
            gripper_dir = os.path.join(split_path, gripper_name)
            if not os.path.isdir(gripper_dir):
                continue

            # Normalize gripper name
            mapped_gripper = cfg.dataset.robot_name_mapping.get(gripper_name, gripper_name.lower())

            if mapped_gripper not in grasp_index:
                grasp_index[mapped_gripper] = {}

            for json_file in os.listdir(gripper_dir):
                if not json_file.endswith(".json"):
                    continue

                try:
                    file_path = os.path.join(gripper_dir, json_file)
                    with open(file_path, "r") as f:
                        grasp_data = json.load(f)
                except Exception as e:
                    print(file_path)
                    raise e

                scene_id = grasp_data.get("scene_id", json_file.replace(".json", ""))

                # Skip if scene wasn't processed successfully
                if scene_id not in scene_data:
                    continue

                if scene_id not in grasp_index[mapped_gripper]:
                    grasp_index[mapped_gripper][scene_id] = {}

                # Process grasps for each object
                obj_id = grasp_data.get("object_id", "unknown")

                if obj_id not in grasp_index[mapped_gripper][scene_id]:
                    grasp_index[mapped_gripper][scene_id][obj_id] = []

                # Load object mesh (cached)
                if obj_id not in object_mesh_cache:
                    object_mesh_cache[obj_id] = load_object_mesh(obj_id, object_dir)
                obj_mesh = object_mesh_cache[obj_id]

                # Sample object points (cached in local frame)
                if obj_id not in object_points_cache and obj_mesh is not None:
                    object_points_cache[obj_id] = obj_mesh.sample(2048)
                obj_points_local = object_points_cache.get(obj_id, None)

                # Get scene pose info (cached)
                if scene_id not in scene_pose_cache:
                    scene_json_path = os.path.join(
                        to_absolute_path(cfg.dataset.dataset_basedir),
                        "scenes",
                        scene_id,
                        f"{scene_id}.json",
                    )
                    if os.path.exists(scene_json_path):
                        with open(scene_json_path, "r") as f:
                            scene_pose_cache[scene_id] = json.load(f)
                    else:
                        scene_pose_cache[scene_id] = {}
                scene_pose_data = scene_pose_cache[scene_id]

                # Get object world transform
                if obj_id in scene_pose_data:
                    target_obj_world_pos = np.array(
                        scene_pose_data[obj_id]["position"], dtype=np.float32
                    )
                    target_obj_world_rot = np.array(
                        scene_pose_data[obj_id]["orientation"], dtype=np.float32
                    )
                    target_obj_world_rot_mx = tm.transformations.quaternion_matrix(
                        target_obj_world_rot
                    )[:3, :3].astype(np.float32)
                else:
                    # Identity transform if pose not available
                    target_obj_world_pos = np.zeros(3, dtype=np.float32)
                    target_obj_world_rot_mx = np.eye(3, dtype=np.float32)

                # Transform object points to world frame
                obj_points_world = None
                if obj_points_local is not None:
                    obj_points_world = (
                        target_obj_world_rot_mx @ obj_points_local.T
                    ).T + target_obj_world_pos

                # Process each grasp
                num_grasps = len(grasp_data.get("pose", []))
                stable_poses = grasp_data.get("stable_poses", grasp_data.get("pose", []))
                stable_dofs = grasp_data.get("stable_dofs", grasp_data.get("dofs", []))

                # Get table bounds from config
                # Convert to numpy for faster processing
                stable_poses_np = np.array(stable_poses, dtype=np.float32)
                stable_dofs_np = np.array(stable_dofs, dtype=np.float32)
                dof_indices = cfg.dataset.dof_mapping[mapped_gripper]

                for i in range(min(num_grasps, len(stable_poses), len(stable_dofs))):
                    grasp_pose_np = stable_poses_np[i]
                    grasp_dofs_np = stable_dofs_np[i][dof_indices]

                    # Convert grasp pose to world frame (using numpy)
                    grasp_position = grasp_pose_np[:3]
                    grasp_rotation_quat = grasp_pose_np[3:7]

                    # Convert quaternion to rotation matrix
                    grasp_rotation_mx = tm.transformations.quaternion_matrix(grasp_rotation_quat)[
                        :3, :3
                    ]
                    grasp_rotation_mx_world = target_obj_world_rot_mx @ grasp_rotation_mx

                    # Convert back to quaternion
                    grasp_rotation_quat_world = tm.transformations.quaternion_from_matrix(
                        np.vstack(
                            [np.hstack([grasp_rotation_mx_world, np.zeros((3, 1))]), [0, 0, 0, 1]]
                        )
                    ).astype(np.float32)

                    grasp_position_world = (
                        target_obj_world_rot_mx @ grasp_position + target_obj_world_pos
                    )
                    grasp_position_offset_world = (
                        target_obj_world_rot_mx
                        @ (
                            grasp_position
                            + grasp_rotation_mx @ cfg.dataset.base_pt_offset[mapped_gripper]
                        )
                        + target_obj_world_pos
                    )

                    # Filter out grasps outside table boundaries (XY plane)
                    min_xy = cfg.dataset.spatial_bounds.min_xy
                    max_xy = cfg.dataset.spatial_bounds.max_xy
                    gx, gy = grasp_position_world[0], grasp_position_world[1]
                    if not (min_xy <= gx <= max_xy and min_xy <= gy <= max_xy):
                        continue  # Skip this grasp

                    # Combine into q vector: position + rot6d + dofs (using numpy)
                    rot6d = get_rot6d_from_quat(grasp_rotation_quat_world)
                    grasp_q = np.concatenate([grasp_position_world, rot6d, grasp_dofs_np])

                    # Find the SEED_PT_CANDIDATES closest object points to the
                    # gripper's palm centre. The dataloader draws one of them per
                    # sample as the ground-truth seed point, so store them sorted
                    # nearest-first (index 0 is the deterministic choice used for
                    # validation).
                    closest_pts_world = None
                    if obj_points_world is not None:
                        distances = np.linalg.norm(
                            obj_points_world - grasp_position_offset_world, axis=1
                        )

                        k = min(SEED_PT_CANDIDATES, len(obj_points_world))
                        closest_indices = np.argpartition(distances, k - 1)[:k]
                        closest_indices = closest_indices[np.argsort(distances[closest_indices])]

                        closest_pts_world = obj_points_world[closest_indices]

                    grasp_index[mapped_gripper][scene_id][obj_id].append(
                        {
                            "gripper_id": mapped_gripper,
                            "object_id": obj_id,
                            "scene_id": scene_id,
                            "q": torch.from_numpy(grasp_q).float(),
                            "closest_object_points": (
                                torch.from_numpy(closest_pts_world).float()
                                if closest_pts_world is not None
                                else None
                            ),
                        }
                    )

                # Remove objects with no grasps
                empty_objects = [
                    o for o, gs in grasp_index[mapped_gripper][scene_id].items() if not gs
                ]
                for o in empty_objects:
                    del grasp_index[mapped_gripper][scene_id][o]

            # Remove scenes with no objects
            empty_scenes = [s for s, objs in grasp_index[mapped_gripper].items() if not objs]
            for s in empty_scenes:
                del grasp_index[mapped_gripper][s]

        # Remove grippers with no scenes
        empty_grippers = [g for g, scenes in grasp_index.items() if not scenes]
        for g in empty_grippers:
            del grasp_index[g]

        return grasp_index

    # Process train and val splits
    train_grasps = process_grasps_for_split(cfg.dataset.scene_train_dir, "train")
    val_grasps = process_grasps_for_split(cfg.dataset.scene_val_dir, "val")

    # Save grasp data
    grasp_data_path = os.path.join(
        to_absolute_path(cfg.dataset.dataset_basedir), "rgbd_scene_grasp_data.pt"
    )
    torch.save(
        {
            "train": train_grasps,
            "val": val_grasps,
        },
        grasp_data_path,
    )
    print(f"Grasp data saved to {grasp_data_path}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("Preprocessing Complete!")
    print("=" * 50)
    print(f"Robots processed: {len(data_dict)}")
    print(f"Scenes processed: {len(scene_data)}")
    print(f"Train grippers: {list(train_grasps.keys())}")
    print(f"Val grippers: {list(val_grasps.keys())}")

    total_train_grasps = sum(
        len(grasps)
        for gripper_scenes in train_grasps.values()
        for scene_objs in gripper_scenes.values()
        for grasps in scene_objs.values()
    )
    total_val_grasps = sum(
        len(grasps)
        for gripper_scenes in val_grasps.values()
        for scene_objs in gripper_scenes.values()
        for grasps in scene_objs.values()
    )
    print(f"Total train grasps: {total_train_grasps}")
    print(f"Total val grasps: {total_val_grasps}")


if __name__ == "__main__":
    main()
