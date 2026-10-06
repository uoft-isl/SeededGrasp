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
Dataloader for RGBD scene dataset with hierarchical sampling.

Sampling strategy:
    1. Sample a random gripper (from available grippers)
    2. Sample a random scene (from scenes with grasps for that gripper)
    3. Sample a random object (with non-zero grasps in that scene)
    4. Sample a random grasp for that object

This hierarchical approach ensures balanced sampling across grippers and objects,
avoiding bias towards grippers/scenes with more grasps.

Usage with train_flow.py:
    Add a new branch for "gnndatasetrgbdscene" in the dataloader handling:

    elif "gnndatasetrgbdscene" in cfg.dataset.dataloader_name:
        (
            _,  # field_names
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
        ) = data
        q = q.to(device)
        obj_adj = scene_adj.to(device)  # Use scene as "object" context
        obj_features = scene_features.to(device)
        robot_name = gripper_id
        # ... rest of processing
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils import data

from utils import math_utils

_DEFAULT_CANONICAL_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _normalize_gripper_name(gripper_id: str) -> str:
    return str(gripper_id).lower()


def build_gripper_alignment_transforms(
    gripper_alignment_rot6d: Optional[Dict[str, List[float]]] = None,
    canonical_rot6d: Optional[List[float]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Build per-gripper right-multiplication transforms in SO(3).

    Given world-frame rotation matrix R and per-gripper base frame B, aligned
    training frame uses R_aligned = R @ (B^T @ C), where C is canonical frame.
    Inverse for inference/export is R_original = R_aligned @ (C^T @ B).
    """
    transforms: Dict[str, Dict[str, torch.Tensor]] = {}
    if not gripper_alignment_rot6d:
        return transforms

    canonical = torch.tensor(
        canonical_rot6d if canonical_rot6d is not None else _DEFAULT_CANONICAL_ROT6D,
        dtype=torch.float32,
    ).view(1, 6)
    if canonical.shape[1] != 6:
        raise ValueError("canonical_rot6d must have exactly 6 values")
    canonical_mat = math_utils.robust_compute_rotation_matrix_from_ortho6d(canonical)[0]

    for gripper_name, rot6d in gripper_alignment_rot6d.items():
        base_rot6d = torch.tensor(rot6d, dtype=torch.float32).view(1, 6)
        if base_rot6d.shape[1] != 6:
            raise ValueError(f"Alignment rot6d for gripper '{gripper_name}' must have 6 values")
        base_mat = math_utils.robust_compute_rotation_matrix_from_ortho6d(base_rot6d)[0]
        align_right = base_mat.transpose(0, 1) @ canonical_mat
        inverse_right = canonical_mat.transpose(0, 1) @ base_mat
        transforms[_normalize_gripper_name(gripper_name)] = {
            "align_right": align_right,
            "inverse_right": inverse_right,
        }

    return transforms


def apply_gripper_rotation_alignment_with_transforms(
    q: torch.Tensor,
    gripper_id: str,
    transforms: Dict[str, Dict[str, torch.Tensor]],
    inverse: bool = False,
    warned_unknown_grippers: Optional[set] = None,
) -> torch.Tensor:
    """Apply forward/inverse per-gripper rotation-frame alignment on q[:, 3:9]."""
    if q.shape[-1] < 9:
        raise ValueError("q must contain at least 9 values: [xyz, rot6d]")

    gripper_key = _normalize_gripper_name(gripper_id)
    transform = transforms.get(gripper_key)
    if transform is None:
        if warned_unknown_grippers is not None and gripper_key not in warned_unknown_grippers:
            print(
                f"WARNING: No rotation alignment configured for gripper '{gripper_id}'. Using identity."
            )
            warned_unknown_grippers.add(gripper_key)
        return q

    right_mult = transform["inverse_right" if inverse else "align_right"]
    q_was_1d = q.ndim == 1
    q_in = q.unsqueeze(0) if q_was_1d else q
    q_out = q_in.clone()

    rot_mat = math_utils.robust_compute_rotation_matrix_from_ortho6d(q_in[:, 3:9])
    right_mult = (
        right_mult.to(device=q_in.device, dtype=q_in.dtype)
        .unsqueeze(0)
        .expand(q_in.shape[0], -1, -1)
    )
    aligned_rot = rot_mat @ right_mult
    q_out[:, 3:9] = aligned_rot[:, :, :2].transpose(1, 2).reshape(q_in.shape[0], 6)

    return q_out.squeeze(0) if q_was_1d else q_out


def apply_gripper_rotation_alignment(
    q: torch.Tensor,
    gripper_id: str,
    gripper_alignment_rot6d: Optional[Dict[str, List[float]]] = None,
    canonical_rot6d: Optional[List[float]] = None,
    inverse: bool = False,
    warned_unknown_grippers: Optional[set] = None,
) -> torch.Tensor:
    """Convenience wrapper for one-off alignment calls."""
    transforms = build_gripper_alignment_transforms(gripper_alignment_rot6d, canonical_rot6d)
    return apply_gripper_rotation_alignment_with_transforms(
        q=q,
        gripper_id=gripper_id,
        transforms=transforms,
        inverse=inverse,
        warned_unknown_grippers=warned_unknown_grippers,
    )


def _maybe_align_grasp_q(
    grasp: Dict[str, Any],
    gripper_id: str,
    transforms: Dict[str, Dict[str, torch.Tensor]],
    warned_unknown_grippers: set,
) -> Dict[str, Any]:
    """Return grasp with canonicalized rot6d if alignment config is provided."""
    q = grasp.get("q", None)
    if q is None:
        return grasp
    if not transforms:
        return grasp

    aligned_q = apply_gripper_rotation_alignment_with_transforms(
        q=q,
        gripper_id=gripper_id,
        transforms=transforms,
        inverse=False,
        warned_unknown_grippers=warned_unknown_grippers,
    )
    grasp = grasp.copy()
    grasp["q"] = aligned_q
    return grasp


def _pick_seed_pt(candidate_points, random_choice: bool):
    """Pick one ground-truth seed point from the stored nearest-neighbour set.

    Preprocessing stores the SEED_PT_CANDIDATES object points closest to the
    gripper's palm centre, sorted nearest-first. Training draws one uniformly
    at random each time a grasp is sampled, so the network sees a spread of
    plausible seed points for the same grasp instead of a single fixed one.
    Validation always takes the nearest point so the split stays deterministic.

    Returns a (1, 3) tensor, or None when the grasp has no stored candidates.
    """
    if candidate_points is None:
        return None
    if candidate_points.shape[0] <= 1:
        return candidate_points
    idx = torch.randint(candidate_points.shape[0], (1,)).item() if random_choice else 0
    return candidate_points[idx : idx + 1]


class GNNDatasetRGBDScene(data.Dataset):
    """
    Dataloader for RGBD scene-based grasping dataset.

    Implements hierarchical sampling: gripper -> scene -> object -> grasp
    to ensure balanced representation across all levels.
    """

    def __init__(
        self,
        dataset_basedir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: str = "train",
        robot_name_list: Optional[List[str]] = None,
        pad_q: int = 0,
        epoch_multiplier: int = 10,
        gripper_alignment_rot6d: Optional[Dict[str, List[float]]] = None,
        canonical_rot6d: Optional[List[float]] = None,
        random_seed_pt: bool = True,
    ):
        """
        Initialize the RGBD scene dataset.

        Args:
            dataset_basedir: Base directory containing preprocessed data
            device: Device to use ('cuda' or 'cpu')
            mode: Dataset mode ('train', 'validate', or 'full')
            robot_name_list: List of robot names to include (None = all)
            pad_q: If > 0, pad q vectors to this length with NaN
            epoch_multiplier: Virtual dataset size multiplier for epoch length
        """
        self.device = device
        self.dataset_basedir = dataset_basedir
        self.mode = mode
        self.pad_q = pad_q
        self.epoch_multiplier = epoch_multiplier
        self.random_seed_pt = random_seed_pt
        self._warned_unknown_grippers = set()
        self._alignment_transforms = build_gripper_alignment_transforms(
            gripper_alignment_rot6d=gripper_alignment_rot6d,
            canonical_rot6d=canonical_rot6d,
        )

        # Load robot point cloud data
        print("Loading robot point clouds and adjacency matrices...")
        self.robot_pc_adj = torch.load(
            os.path.join(dataset_basedir, "gnn_robot_adj_point_clouds_new.pt"), weights_only=False
        )

        # Convert sparse to dense for PyTorch 2.0 compatibility and pad rest_pose
        for robot_name in self.robot_pc_adj.keys():
            robo_adj, robo_features, rest_pose, rname = self.robot_pc_adj[robot_name]
            if robo_adj.is_sparse:
                robo_adj = robo_adj.to_dense()
            # Pad rest_pose if needed
            if pad_q > 0:
                rp = rest_pose.squeeze(0)  # (D,)
                if rp.shape[0] < pad_q:
                    rest_pose = torch.cat(
                        [rp, torch.full((pad_q - rp.shape[0],), float("nan"))], dim=0
                    ).unsqueeze(0)
            self.robot_pc_adj[robot_name] = (robo_adj, robo_features, rest_pose, rname)

        # Load scene point cloud data
        print("Loading scene point clouds and adjacency matrices...")
        self.scene_pc_adj = torch.load(
            os.path.join(dataset_basedir, "gnn_scene_adj_point_clouds_new.pt"), weights_only=False
        )

        # Keep scene_adj as sparse to save memory - will convert to dense on GPU when needed
        # No conversion needed here

        # Load grasp data
        print("Loading grasp data...")
        grasp_data = torch.load(
            os.path.join(dataset_basedir, "rgbd_scene_grasp_data.pt"), weights_only=False
        )

        # Select appropriate split
        if mode == "train":
            self.grasp_index = grasp_data.get("train", {})
        elif mode in ["validate", "val"]:
            self.grasp_index = grasp_data.get("val", {})
        elif mode == "full":
            # Merge train and val
            self.grasp_index = self._merge_grasp_indices(
                grasp_data.get("train", {}), grasp_data.get("val", {})
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'train', 'validate', or 'full'")

        # Filter by robot_name_list if specified
        if robot_name_list:
            self.grasp_index = {k: v for k, v in self.grasp_index.items() if k in robot_name_list}

        # Build sampling indices for O(1) random access
        self._build_sampling_indices()

        print(f"Dataset initialized with {len(self.gripper_list)} grippers")
        print(f"Total scenes: {self.total_scenes}")
        print(f"Total object instances: {self.total_objects}")
        print(f"Total grasps: {self.total_grasps}")

    def _merge_grasp_indices(self, train_idx: Dict, val_idx: Dict) -> Dict:
        """Merge train and val grasp indices."""
        merged = {}

        for gripper in set(train_idx.keys()) | set(val_idx.keys()):
            merged[gripper] = {}

            for source in [train_idx, val_idx]:
                if gripper not in source:
                    continue

                for scene_id, objects in source[gripper].items():
                    if scene_id not in merged[gripper]:
                        merged[gripper][scene_id] = {}

                    for obj_id, grasps in objects.items():
                        if obj_id not in merged[gripper][scene_id]:
                            merged[gripper][scene_id][obj_id] = []
                        merged[gripper][scene_id][obj_id].extend(grasps)

        return merged

    def _build_sampling_indices(self):
        """
        Build FLAT indices for O(1) random access sampling.

        Pre-flattens the entire hierarchy into a list for fast random indexing,
        avoiding dictionary lookups and recursive sampling at runtime.
        """
        self.gripper_list = list(self.grasp_index.keys())
        self.gripper_to_scenes: Dict[str, List[str]] = {}
        self.scene_to_objects: Dict[Tuple[str, str], List[str]] = {}

        self.total_scenes = 0
        self.total_objects = 0
        self.total_grasps = 0

        # Flatten all grasps into a list for O(1) random access
        # Each entry: (gripper_id, scene_id, object_id, grasp_data)
        self._flat_grasps = []

        for gripper in self.gripper_list:
            scenes = list(self.grasp_index[gripper].keys())
            self.gripper_to_scenes[gripper] = scenes
            self.total_scenes += len(scenes)

            for scene in scenes:
                objects = list(self.grasp_index[gripper][scene].keys())
                # Only include objects with non-zero grasps
                objects = [obj for obj in objects if len(self.grasp_index[gripper][scene][obj]) > 0]
                self.scene_to_objects[(gripper, scene)] = objects
                self.total_objects += len(objects)

                for obj in objects:
                    grasps = self.grasp_index[gripper][scene][obj]
                    self.total_grasps += len(grasps)
                    # Add each grasp to flat list with pre-padded q
                    for grasp in grasps:
                        # Pre-pad q if needed
                        if self.pad_q > 0:
                            q = grasp["q"]
                            if q.shape[0] < self.pad_q:
                                grasp = grasp.copy()  # Don't modify original
                                grasp["q"] = torch.cat(
                                    [q, torch.full((self.pad_q - q.shape[0],), float("nan"))], dim=0
                                )

                        grasp = _maybe_align_grasp_q(
                            grasp=grasp,
                            gripper_id=gripper,
                            transforms=self._alignment_transforms,
                            warned_unknown_grippers=self._warned_unknown_grippers,
                        )

                        self._flat_grasps.append((gripper, scene, obj, grasp))

        # Convert gripper_list to numpy array for fast indexing
        self._num_grasps = len(self._flat_grasps)
        print(f"Built flat index with {self._num_grasps} grasps for O(1) sampling")

    def __len__(self) -> int:
        """
        Return virtual dataset length.

        Since we sample randomly, the actual length is arbitrary.
        We use epoch_multiplier * number of unique objects to define epoch length.
        """
        return max(1, self.total_objects * self.epoch_multiplier)

    def sample_grasp(self, idx: int = None) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        O(1) random grasp sampling from pre-flattened index.

        Args:
            idx: Optional index to use. If None, samples randomly.

        Returns:
            Tuple of (gripper_id, scene_id, object_id, grasp_data)
        """
        if idx is None:
            idx = np.random.randint(0, self._num_grasps)
        else:
            idx = idx % self._num_grasps
        return self._flat_grasps[idx]

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single training sample with O(1) random access.

        Uses idx to deterministically select from flattened grasp list,
        ensuring reproducibility when using seeded random.

        Returns:
            Tuple containing:
                - field_names: List of field names for reference
                - q: Grasp configuration (position + rot6d + dofs)
                - robot_adj: Robot adjacency matrix (None - not used)
                - robot_features: Robot point cloud features (None - not used)
                - rest_pose: Robot rest pose
                - scene_adj: Scene adjacency matrix
                - scene_features: Scene point cloud features
                - gripper_id: Gripper identifier
                - object_id: Object identifier
                - scene_id: Scene identifier
                - closest_object_points: one seed point (1, 3) drawn from the
                  stored nearest-neighbour candidates (world frame)
        """
        # Use idx for deterministic sampling (wraps around if idx > num_grasps)
        gripper_id, scene_id, object_id, grasp = self.sample_grasp(idx)

        # Get robot data - only fetch rest_pose (robot_adj/robot_features not used in model)
        _, _, rest_pose, _ = self.robot_pc_adj[gripper_id]

        # Get scene data
        scene_data = self.scene_pc_adj[scene_id]
        scene_adj = scene_data["scene_adj"]
        scene_features = scene_data["scene_features"]

        # Get grasp configuration (already padded during init if pad_q > 0)
        q = grasp["q"]
        closest_object_points = self._select_seed_pt(grasp.get("closest_object_points", None))

        field_names = [
            "q",
            "robot_adj",
            "robot_features",
            "rest_pose",
            "scene_adj",
            "scene_features",
            "gripper_id",
            "object_id",
            "scene_id",
            "closest_object_points",
        ]

        return (
            field_names,
            q,
            None,  # robot_adj - not used, skip to save collation time
            None,  # robot_features - not used, skip to save collation time
            rest_pose,
            scene_adj,
            scene_features,
            gripper_id,
            object_id,
            scene_id,
            closest_object_points,
        )

    def _select_seed_pt(self, candidate_points):
        return _pick_seed_pt(candidate_points, self.random_seed_pt)

    def get_robot_data(self, gripper_id: str) -> Tuple:
        """Get robot point cloud data for a specific gripper."""
        return self.robot_pc_adj[gripper_id]

    def get_scene_data(self, scene_id: str) -> Dict:
        """Get scene point cloud data for a specific scene."""
        return self.scene_pc_adj[scene_id]

    def get_all_grasps_for_scene(self, gripper_id: str, scene_id: str) -> Dict[str, List[Dict]]:
        """Get all grasps for a given gripper and scene, organized by object."""
        if gripper_id not in self.grasp_index:
            return {}
        if scene_id not in self.grasp_index[gripper_id]:
            return {}
        return self.grasp_index[gripper_id][scene_id]

    @property
    def available_grippers(self) -> List[str]:
        """Return list of available gripper names."""
        return self.gripper_list.copy()

    @property
    def available_scenes(self) -> List[str]:
        """Return list of all available scene IDs."""
        scenes = set()
        for gripper_scenes in self.grasp_index.values():
            scenes.update(gripper_scenes.keys())
        return list(scenes)


class GNNDatasetRGBDSceneWeighted(GNNDatasetRGBDScene):
    """
    Variant with weighted sampling to handle imbalanced gripper distributions.

    Allows specifying sampling weights per gripper, which is useful when
    some grippers have significantly more grasps than others.
    """

    def __init__(
        self,
        dataset_basedir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: str = "train",
        robot_name_list: Optional[List[str]] = None,
        pad_q: int = 0,
        epoch_multiplier: int = 10,
        gripper_weights: Optional[Dict[str, float]] = None,
        gripper_alignment_rot6d: Optional[Dict[str, List[float]]] = None,
        canonical_rot6d: Optional[List[float]] = None,
        random_seed_pt: bool = True,
    ):
        """
        Initialize with optional gripper sampling weights.

        Args:
            gripper_weights: Dict mapping gripper_id -> sampling weight.
                            If None, uses uniform weights.
        """
        super().__init__(
            dataset_basedir=dataset_basedir,
            device=device,
            mode=mode,
            robot_name_list=robot_name_list,
            pad_q=pad_q,
            epoch_multiplier=epoch_multiplier,
            gripper_alignment_rot6d=gripper_alignment_rot6d,
            canonical_rot6d=canonical_rot6d,
            random_seed_pt=random_seed_pt,
        )

        # Set up weighted sampling
        if gripper_weights is None:
            self.gripper_weights = {g: 1.0 for g in self.gripper_list}
        else:
            self.gripper_weights = {g: gripper_weights.get(g, 1.0) for g in self.gripper_list}

        # Normalize weights
        total_weight = sum(self.gripper_weights.values())
        self.gripper_weights = {g: w / total_weight for g, w in self.gripper_weights.items()}

        # Build per-gripper flat indices for weighted sampling
        self._gripper_to_flat_indices = {}
        for i, (gripper, scene, obj, grasp) in enumerate(self._flat_grasps):
            if gripper not in self._gripper_to_flat_indices:
                self._gripper_to_flat_indices[gripper] = []
            self._gripper_to_flat_indices[gripper].append(i)

        # Convert to numpy arrays for fast sampling
        for gripper in self._gripper_to_flat_indices:
            self._gripper_to_flat_indices[gripper] = np.array(
                self._gripper_to_flat_indices[gripper], dtype=np.int64
            )

        # Pre-compute gripper sampling arrays
        self._gripper_arr = np.array(list(self.gripper_weights.keys()))
        self._gripper_probs = np.array([self.gripper_weights[g] for g in self._gripper_arr])

    def sample_grasp(self, idx: int = None) -> Tuple[str, str, str, Dict[str, Any]]:
        """Sample with weighted gripper selection using O(1) lookups."""
        # 1. Weighted sample of gripper
        gripper = np.random.choice(self._gripper_arr, p=self._gripper_probs)

        # 2. Random sample from that gripper's flattened grasps
        gripper_indices = self._gripper_to_flat_indices[gripper]
        flat_idx = gripper_indices[np.random.randint(len(gripper_indices))]

        return self._flat_grasps[flat_idx]


class GNNDatasetRGBDSceneHierarchical(data.Dataset):
    """
    Dataloader with true hierarchical uniform sampling.

    At each level the choice is uniform over the available options:
        1. Uniform sample a gripper  (from all grippers in the split)
        2. Uniform sample a scene    (from scenes that have grasps for that gripper)
        3. Uniform sample an object  (from objects with ≥1 grasp in that scene/gripper)
        4. Uniform sample a grasp    (from grasps for that object)

    This guarantees that every gripper is seen equally often regardless of how
    many scenes/objects/grasps it has, and likewise at every subsequent level.
    """

    def __init__(
        self,
        dataset_basedir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mode: str = "train",
        train_scene_percent: float = 100.0,
        scene_subset_seed: Optional[int] = None,
        robot_name_list: Optional[List[str]] = None,
        pad_q: int = 0,
        epoch_multiplier: int = 10,
        gripper_alignment_rot6d: Optional[Dict[str, List[float]]] = None,
        canonical_rot6d: Optional[List[float]] = None,
        random_seed_pt: bool = True,
    ):
        self.device = device
        self.dataset_basedir = dataset_basedir
        self.mode = mode
        self.train_scene_percent = train_scene_percent
        self.scene_subset_seed = scene_subset_seed
        self.pad_q = pad_q
        self.epoch_multiplier = epoch_multiplier
        self.random_seed_pt = random_seed_pt
        self._warned_unknown_grippers = set()
        self._alignment_transforms = build_gripper_alignment_transforms(
            gripper_alignment_rot6d=gripper_alignment_rot6d,
            canonical_rot6d=canonical_rot6d,
        )

        # ----- Load shared data (robot PCs, scene PCs) -----
        print("Loading robot point clouds and adjacency matrices...")
        self.robot_pc_adj = torch.load(
            os.path.join(dataset_basedir, "gnn_robot_adj_point_clouds_new.pt"),
            weights_only=False,
        )
        for robot_name in self.robot_pc_adj:
            robo_adj, robo_features, rest_pose, rname = self.robot_pc_adj[robot_name]
            if robo_adj.is_sparse:
                robo_adj = robo_adj.to_dense()
            # Pad rest_pose if needed
            if pad_q > 0:
                rp = rest_pose.squeeze(0)  # (D,)
                if rp.shape[0] < pad_q:
                    rest_pose = torch.cat(
                        [rp, torch.full((pad_q - rp.shape[0],), float("nan"))], dim=0
                    ).unsqueeze(0)
            self.robot_pc_adj[robot_name] = (robo_adj, robo_features, rest_pose, rname)

        print("Loading scene point clouds and adjacency matrices...")
        self.scene_pc_adj = torch.load(
            os.path.join(dataset_basedir, "gnn_scene_adj_point_clouds_new.pt"),
            weights_only=False,
        )
        # Keep scene_adj as sparse to save memory - will convert to dense on GPU when needed
        # No conversion needed here

        # ----- Load & select grasp split -----
        print("Loading grasp data...")
        grasp_data = torch.load(
            os.path.join(dataset_basedir, "rgbd_scene_grasp_data.pt"),
            weights_only=False,
        )

        if mode == "train":
            raw_index = grasp_data.get("train", {})
        elif mode in ("validate", "val"):
            raw_index = grasp_data.get("val", {})
        elif mode == "full":
            raw_index = self._merge(grasp_data.get("train", {}), grasp_data.get("val", {}))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Filter by robot_name_list
        if robot_name_list:
            raw_index = {k: v for k, v in raw_index.items() if k in robot_name_list}

        if mode == "train" and train_scene_percent < 100.0:
            if train_scene_percent <= 0.0 or train_scene_percent > 100.0:
                raise ValueError("train_scene_percent must be in the range (0, 100].")

            all_scene_ids = self._collect_unique_scene_ids(raw_index)
            if not all_scene_ids:
                raise ValueError("No training scenes available before scene subsampling.")

            num_scenes_to_keep = max(
                1, int(np.ceil(len(all_scene_ids) * (train_scene_percent / 100.0)))
            )
            num_scenes_to_keep = min(num_scenes_to_keep, len(all_scene_ids))

            rng = np.random.default_rng(scene_subset_seed)
            selected_scene_indices = rng.choice(
                len(all_scene_ids), size=num_scenes_to_keep, replace=False
            )
            selected_scene_ids = {all_scene_ids[i] for i in np.atleast_1d(selected_scene_indices)}
            raw_index = self._filter_index_to_scene_ids(raw_index, selected_scene_ids)

            print(
                f"Using {len(selected_scene_ids)} / {len(all_scene_ids)} training scenes "
                f"({train_scene_percent:.1f}%)."
            )

        # ----- Build hierarchy as nested lists (fast numpy indexing) -----
        # _hierarchy[g] = list of scenes
        # _scene_objs[(g, s)] = list of objects
        # _obj_grasps[(g, s, o)] = list of grasp dicts
        self._hierarchy: List[str] = []  # gripper names
        self._gripper_scenes: Dict[str, List[str]] = {}
        self._scene_objs: Dict[Tuple[str, str], List[str]] = {}
        self._obj_grasps: Dict[Tuple[str, str, str], List[Dict]] = {}

        total_scenes = 0
        total_objects = 0
        total_grasps = 0

        for gripper, scenes in raw_index.items():
            valid_scenes: List[str] = []
            for scene_id, objects in scenes.items():
                valid_objs: List[str] = []
                for obj_id, grasps in objects.items():
                    if grasps:
                        valid_objs.append(obj_id)
                        # Pre-pad q values if needed
                        if self.pad_q > 0:
                            padded_grasps = []
                            for grasp in grasps:
                                q = grasp["q"]
                                if q.shape[0] < self.pad_q:
                                    grasp = grasp.copy()  # Don't modify original
                                    grasp["q"] = torch.cat(
                                        [q, torch.full((self.pad_q - q.shape[0],), float("nan"))],
                                        dim=0,
                                    )
                                grasp = _maybe_align_grasp_q(
                                    grasp=grasp,
                                    gripper_id=gripper,
                                    transforms=self._alignment_transforms,
                                    warned_unknown_grippers=self._warned_unknown_grippers,
                                )
                                padded_grasps.append(grasp)
                            grasps = padded_grasps
                        elif self._alignment_transforms:
                            aligned_grasps = []
                            for grasp in grasps:
                                aligned_grasps.append(
                                    _maybe_align_grasp_q(
                                        grasp=grasp,
                                        gripper_id=gripper,
                                        transforms=self._alignment_transforms,
                                        warned_unknown_grippers=self._warned_unknown_grippers,
                                    )
                                )
                            grasps = aligned_grasps

                        self._obj_grasps[(gripper, scene_id, obj_id)] = list(grasps)
                        total_grasps += len(grasps)
                if valid_objs:
                    valid_scenes.append(scene_id)
                    self._scene_objs[(gripper, scene_id)] = valid_objs
                    total_objects += len(valid_objs)
            if valid_scenes:
                self._hierarchy.append(gripper)
                self._gripper_scenes[gripper] = valid_scenes
                total_scenes += len(valid_scenes)

        self.total_grippers = len(self._hierarchy)
        self.total_scenes = total_scenes
        self.total_objects = total_objects
        self.total_grasps = total_grasps

        if self.total_grippers == 0 or self.total_grasps == 0:
            raise ValueError("No grasps remain after applying the configured filters.")

        print(
            f"Hierarchical dataset: {self.total_grippers} grippers, "
            f"{total_scenes} scenes, {total_objects} objects, {total_grasps} grasps"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _merge(a: Dict, b: Dict) -> Dict:
        merged: Dict = {}
        for gripper in set(a) | set(b):
            merged[gripper] = {}
            for src in (a, b):
                if gripper not in src:
                    continue
                for sid, objs in src[gripper].items():
                    if sid not in merged[gripper]:
                        merged[gripper][sid] = {}
                    for oid, gs in objs.items():
                        merged[gripper][sid].setdefault(oid, []).extend(gs)
        return merged

    @staticmethod
    def _collect_unique_scene_ids(index: Dict) -> List[str]:
        scene_ids = set()
        for scenes in index.values():
            scene_ids.update(scenes.keys())
        return sorted(scene_ids)

    @staticmethod
    def _filter_index_to_scene_ids(index: Dict, allowed_scene_ids: set) -> Dict:
        filtered: Dict = {}
        for gripper, scenes in index.items():
            kept_scenes = {
                scene_id: objects
                for scene_id, objects in scenes.items()
                if scene_id in allowed_scene_ids
            }
            if kept_scenes:
                filtered[gripper] = kept_scenes
        return filtered

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return max(1, self.total_objects * self.epoch_multiplier)

    def __getitem__(self, idx: int) -> Tuple:
        # 1. Uniform over grippers
        g_idx = np.random.randint(self.total_grippers)
        gripper_id = self._hierarchy[g_idx]

        # 2. Uniform over scenes for that gripper
        scenes = self._gripper_scenes[gripper_id]
        scene_id = scenes[np.random.randint(len(scenes))]

        # 3. Uniform over objects in that (gripper, scene)
        objs = self._scene_objs[(gripper_id, scene_id)]
        object_id = objs[np.random.randint(len(objs))]

        # 4. Uniform over grasps for that (gripper, scene, object)
        grasps = self._obj_grasps[(gripper_id, scene_id, object_id)]
        grasp = grasps[np.random.randint(len(grasps))]

        # ----- Fetch tensors -----
        # Note: robot_adj and robot_features are not used in the model for RGBD scenes,
        # so we return None to avoid expensive stacking during collation
        _, _, rest_pose, _ = self.robot_pc_adj[gripper_id]

        scene_data = self.scene_pc_adj[scene_id]
        scene_adj = scene_data["scene_adj"]
        scene_features = scene_data["scene_features"]

        # q is already padded during init if pad_q > 0
        q = grasp["q"]
        closest_object_points = self._select_seed_pt(grasp.get("closest_object_points", None))

        field_names = []

        return (
            field_names,
            q,
            None,  # robot_adj - not used, skip to save collation time
            None,  # robot_features - not used, skip to save collation time
            rest_pose,
            scene_adj,
            scene_features,
            gripper_id,
            object_id,
            scene_id,
            closest_object_points,
        )

    def _select_seed_pt(self, candidate_points):
        return _pick_seed_pt(candidate_points, self.random_seed_pt)

    # ---- convenience accessors (same interface as GNNDatasetRGBDScene) ----
    def get_robot_data(self, gripper_id: str) -> Tuple:
        return self.robot_pc_adj[gripper_id]

    def get_scene_data(self, scene_id: str) -> Dict:
        return self.scene_pc_adj[scene_id]

    @property
    def available_grippers(self) -> List[str]:
        return list(self._hierarchy)

    @property
    def available_scenes(self) -> List[str]:
        scenes: set = set()
        for g in self._hierarchy:
            scenes.update(self._gripper_scenes[g])
        return list(scenes)


def collate_fn_rgbd(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function for batching RGBD scene data.

    Handles variable-length q vectors and string fields.
    Skips stacking None values (for unused tensors like robot_adj/robot_features).
    Keeps scene_adj as a list of sparse tensors for memory efficiency.
    Returns a tuple matching the expected unpacking in train_flow.py.
    """
    field_names = batch[0][0]

    # Pre-allocate lists
    batch_size = len(batch)
    q_list = [None] * batch_size
    robot_adj_list = [None] * batch_size
    robot_features_list = [None] * batch_size
    rest_pose_list = [None] * batch_size
    scene_adj_list = [None] * batch_size
    scene_features_list = [None] * batch_size
    gripper_ids = [None] * batch_size
    object_ids = [None] * batch_size
    scene_ids = [None] * batch_size
    closest_object_points_list = [None] * batch_size

    for i, item in enumerate(batch):
        (
            _,
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
        ) = item

        q_list[i] = q
        robot_adj_list[i] = robot_adj
        robot_features_list[i] = robot_features
        rest_pose_list[i] = rest_pose
        scene_adj_list[i] = scene_adj  # Keep as sparse tensor
        scene_features_list[i] = scene_features
        gripper_ids[i] = gripper_id
        object_ids[i] = object_id
        scene_ids[i] = scene_id

        closest_object_points_list[i] = closest_object_points

    # Stack seed points - one already drawn per sample in __getitem__ (B, 1, 3)
    batched_seed_pts = (
        torch.stack(closest_object_points_list, dim=0)
        if closest_object_points_list[0] is not None
        else None
    )

    # Return as tuple to match train_flow.py unpacking
    # scene_adj is returned as list of sparse tensors - convert to dense on GPU in train loop
    # batched_seed_pts: one seed point per sample (B, 1, 3), drawn in __getitem__
    # gripper_ids: list of strings - will be converted to indices in training loop on CPU
    return (
        field_names,
        torch.stack(q_list, dim=0),
        torch.stack(robot_adj_list, dim=0) if batch[0][2] is not None else None,
        torch.stack(robot_features_list, dim=0) if batch[0][3] is not None else None,
        torch.cat(rest_pose_list, dim=0),
        scene_adj_list,  # List of sparse tensors
        torch.stack(scene_features_list, dim=0),
        gripper_ids,  # List of strings
        object_ids,
        scene_ids,
        batched_seed_pts,  # (B, 1, 3)
    )
