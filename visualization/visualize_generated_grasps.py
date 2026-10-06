#!/usr/bin/env python3
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
visualize_generated_grasps.py

Visualize predicted grasps from the output of generate_grasps_from_seed_pts.py,
or raw ground-truth grasps (with --ground-truth).

Usage (run from the repo root):

    python visualization/visualize_generated_grasps.py \
        --grasps  generated_grasps/robotiq_3finger_predicted_grasps.json \
        [--grasps  generated_grasps/Allegro_predicted_grasps.json ...] \
        [--ground-truth] \
        [--indices-json /path/to/indices.json] \
        [--indices-base auto|zero|one] \
        [--config-dir configs] \
        [--max-grasps 10] \
        [--filter-gripper robotiq_3finger] \
        [--filter-scene  scene-multiple-0af9f30fcb159cdc] \
        [--filter-object 2_of_Jenga_Classic_Game]

Each (gripper, scene, object) group within a file is shown as a separate
interactive Plotly figure (browser tab).  Close / advance each figure to see
the next one.

Input JSON format  (output of generate_grasps_from_seed_pts.py):
    {
        "scene_id":   ["<scene_id>", ...],
        "object_id":  ["<object_id>", ...],
        "gripper_id": ["<gripper_id>", ...],
        "pred_pose":  [[trans(3) + rot6d(6)], ...],   # world frame
        "pred_dofs":  [[dofs], ...]                    # dof_mapping order
    }

With --ground-truth (raw feasible_grasps.json "pose"/"dofs", unchanged), the
same "pred_pose"/"pred_dofs" fields instead hold:
    "pred_pose":  [[trans(3) + quat_wxyz(4)], ...]     # object-relative frame
    "pred_dofs":  [[dofs], ...]                         # native simulator order

In that case this script converts each grasp into the world-frame
trans(3) + rot6d(6) + dof_mapping-order dofs format expected by the hand
model, mirroring the conversion in create_gnn_dataset_rgbd_scene.py (which
requires looking up the target object's world pose from the scene JSON, so
--scenes-dir / the dataset's scenes/ directory must be available).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
import trimesh as tm

# ---------------------------------------------------------------------------
# Repo root on sys.path so project imports work
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path

from utils.math_utils import get_rot6d_from_quat

pio.renderers.default = "browser"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_cfg(config_dir: str, config_name: str = "config_rgbd_scene"):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def load_json_grasps(path: str) -> dict:
    import json

    with open(path) as f:
        return json.load(f)


def load_indices_json(path: str) -> list:
    """
    Load indices from JSON.

    Supported formats:
      1) [1, 4, 8, ...]
      2) {"indices": [1, 4, 8, ...]}
    """
    import json

    with open(path) as f:
        payload = json.load(f)

    if isinstance(payload, list):
        raw_indices = payload
    elif isinstance(payload, dict) and "indices" in payload:
        raw_indices = payload["indices"]
    else:
        raise ValueError(
            f"Unsupported indices JSON format in '{path}'. "
            "Use a list of ints or {'indices': [...]}"
        )

    indices: list = []
    for i, value in enumerate(raw_indices):
        if isinstance(value, bool):
            raise ValueError(f"Invalid boolean at indices[{i}] in '{path}'")
        try:
            idx = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid index at indices[{i}] in '{path}': {value}") from exc
        indices.append(idx)

    return indices


def _resolve_index_base(index_values: list, num_items: int, index_base: str) -> int:
    """Resolve index base for filtering: 0 for zero-based, 1 for one-based."""
    if index_base == "zero":
        return 0
    if index_base == "one":
        return 1

    # Auto mode heuristic:
    # - If all indices are >= 1 and max index is at least num_items, assume one-based.
    # - Otherwise default to zero-based.
    if index_values and min(index_values) >= 1 and max(index_values) >= num_items:
        return 1
    return 0


def filter_grasp_data_by_indices(
    data: dict, index_values: list, index_base: str, source_name: str
) -> dict:
    """Filter all list fields with grasp-length by selected indices."""
    required_keys = ["pred_pose", "pred_dofs", "gripper_id", "scene_id", "object_id"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in '{source_name}': {missing}")

    num_items = len(data["pred_pose"])
    resolved_base = _resolve_index_base(index_values, num_items, index_base)

    selected: list = []
    seen = set()
    skipped_out_of_range = 0

    for raw_idx in index_values:
        idx = raw_idx - 1 if resolved_base == 1 else raw_idx
        if idx < 0 or idx >= num_items:
            skipped_out_of_range += 1
            continue
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(idx)

    filtered = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) == num_items:
            filtered[key] = [value[i] for i in selected]
        else:
            filtered[key] = value

    kept = len(selected)
    print(
        f"  Applied indices filter ({'1-based' if resolved_base == 1 else '0-based'}): "
        f"kept {kept}/{num_items} entries"
    )
    if skipped_out_of_range:
        print(f"  [WARN] Skipped {skipped_out_of_range} out-of-range indices from filter file.")

    return filtered


def reconstruct_q(pred_pose: list, pred_dofs: list) -> torch.Tensor:
    """Concatenate pred_pose (9-D: trans(3) + rot6d(6)) and pred_dofs into q (1, D)."""
    pose = torch.tensor(pred_pose, dtype=torch.float32)  # (9,)
    if pose.shape != (9,):
        raise ValueError(
            f"Expected a 9-D pose (trans(3)+rot6d(6)) for generated grasps, "
            f"got shape {tuple(pose.shape)}. If this is a raw ground-truth "
            "grasps file, pass --ground-truth."
        )
    dofs = torch.tensor(pred_dofs, dtype=torch.float32)  # (N,)
    return torch.cat([pose, dofs], dim=0).unsqueeze(0)  # (1, 9+N)


def reconstruct_q_ground_truth(
    pred_pose: list,
    pred_dofs: list,
    dof_indices,
    object_world_pos: np.ndarray,
    object_world_quat: np.ndarray,
) -> torch.Tensor:
    """
    Convert a raw ground-truth grasp (object-relative trans(3) + quat_wxyz(4),
    native-simulator dof order) into the world-frame trans(3) + rot6d(6) + dofs
    q format expected by the hand model. Mirrors the conversion in
    create_gnn_dataset_rgbd_scene.py so ground-truth and generated grasps are
    rendered with the same representation.
    """
    pose_np = np.array(pred_pose, dtype=np.float32)
    if pose_np.shape != (7,):
        raise ValueError(
            f"--ground-truth expects a 7-D pose (trans(3)+quat_wxyz(4)), "
            f"got shape {pose_np.shape}. If this is a generated-grasps file, "
            "drop --ground-truth."
        )
    grasp_position = pose_np[:3]
    grasp_rotation_quat = pose_np[3:7]

    grasp_rotation_mx = tm.transformations.quaternion_matrix(grasp_rotation_quat)[:3, :3]
    object_rotation_mx = tm.transformations.quaternion_matrix(object_world_quat)[:3, :3]

    grasp_rotation_mx_world = object_rotation_mx @ grasp_rotation_mx
    grasp_position_world = object_rotation_mx @ grasp_position + object_world_pos

    grasp_rotation_quat_world = tm.transformations.quaternion_from_matrix(
        np.vstack(
            [
                np.hstack([grasp_rotation_mx_world, np.zeros((3, 1))]),
                [0, 0, 0, 1],
            ]
        )
    )
    rot6d = get_rot6d_from_quat(grasp_rotation_quat_world)

    dof_indices_np = np.array(dof_indices, dtype=int)
    dofs_np = np.array(pred_dofs, dtype=np.float32)[dof_indices_np]

    q_np = np.concatenate(
        [
            grasp_position_world.astype(np.float32),
            np.array(rot6d, dtype=np.float32),
            dofs_np,
        ]
    )
    return torch.tensor(q_np, dtype=torch.float32).unsqueeze(0)  # (1, 9+N)


def lookup_object_world_pose(
    scene_id: str,
    object_id: str,
    scenes_dir: Optional[str],
    scene_pose_cache: dict,
) -> Optional[tuple]:
    """Return (position(3,), quat_wxyz(4,)) for an object in a scene, or None if unavailable."""
    if scenes_dir is None:
        return None
    scene_pose_data = _load_scene_pose_json(scene_id, scenes_dir, scene_pose_cache)
    entry = scene_pose_data.get(object_id)
    if entry is None:
        return None
    pos = np.array(entry["position"], dtype=np.float32)
    quat = np.array(entry["orientation"], dtype=np.float32)
    return pos, quat


def _as_scalar_id(value):
    """Normalize string/list-wrapped IDs to a plain scalar value."""
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return value[0]
    return value


def group_grasps(data: dict, group_by: str = "gso"):
    """
    Group flat parallel lists.

    group_by:
      - gso:   (gripper, scene, object)
      - gs:    (gripper, scene)
      - scene: (scene,)
      - all:   one group for everything in a file

    Each group is a list of (pose, dofs, gripper_id, scene_id, object_id)
    tuples. The per-grasp identifiers are kept (not just folded into the group
    key) so that --ground-truth conversion can look up the correct object pose
    and dof_mapping for every grasp even when grouping coarser than "gso"
    mixes multiple objects/grippers/scenes into one group.
    """
    groups: dict = defaultdict(list)
    for pose, dofs, gripper, scene, obj in zip(
        data["pred_pose"],
        data["pred_dofs"],
        data["gripper_id"],
        data["scene_id"],
        data["object_id"],
    ):
        gripper_id = _as_scalar_id(gripper)
        scene_id = _as_scalar_id(scene)
        object_id = _as_scalar_id(obj)

        if group_by == "gso":
            key = (gripper_id, scene_id, object_id)
        elif group_by == "gs":
            key = (gripper_id, scene_id)
        elif group_by == "scene":
            key = (scene_id,)
        elif group_by == "all":
            key = ("ALL", "ALL", "ALL")
        else:
            raise ValueError(f"Unsupported group_by value: {group_by}")

        groups[key].append((pose, dofs, gripper_id, scene_id, object_id))
    return groups


def _expand_group_key(key: tuple):
    """Expand variable-length group keys into (gripper, scene, object)."""
    if len(key) == 3:
        return key
    if len(key) == 2:
        return key[0], key[1], "ALL"
    if len(key) == 1:
        return "ALL", key[0], "ALL"
    return "ALL", "ALL", "ALL"


def _resolve_object_mesh_path(object_id: str, object_dir: str, gso_obj_names: set) -> Optional[str]:
    """Return object mesh path for either GSO or YCB object IDs."""
    if object_id in gso_obj_names:
        path = os.path.join(object_dir, f"GSO/{object_id}/meshes/model.obj")
    else:
        path = os.path.join(object_dir, f"YCB/{object_id}/textured.obj")
    return path if os.path.exists(path) else None


def _load_scene_pose_json(scene_id: str, scenes_dir: str, scene_pose_cache: dict) -> dict:
    """Load and cache per-scene object pose JSON."""
    if scene_id in scene_pose_cache:
        return scene_pose_cache[scene_id]

    scene_json_path = os.path.join(scenes_dir, scene_id, f"{scene_id}.json")
    if not os.path.exists(scene_json_path):
        scene_pose_cache[scene_id] = {}
        return scene_pose_cache[scene_id]

    with open(scene_json_path) as f:
        scene_pose_cache[scene_id] = json.load(f)
    return scene_pose_cache[scene_id]


def _build_object_mesh_trace(
    scene_id: str,
    object_id: str,
    object_dir: str,
    scenes_dir: str,
    gso_obj_names: set,
    object_mesh_cache: dict,
    scene_pose_cache: dict,
) -> Optional[go.Mesh3d]:
    """Build a world-frame Mesh3d trace for the target object, if available."""
    if object_id not in object_mesh_cache:
        mesh_path = _resolve_object_mesh_path(object_id, object_dir, gso_obj_names)
        if mesh_path is None:
            object_mesh_cache[object_id] = None
        else:
            mesh_loaded = tm.load(mesh_path)
            if isinstance(mesh_loaded, tm.Scene):
                geometries = [g for g in mesh_loaded.geometry.values() if isinstance(g, tm.Trimesh)]
                object_mesh_cache[object_id] = (
                    tm.util.concatenate(geometries) if geometries else None
                )
            elif isinstance(mesh_loaded, tm.Trimesh):
                object_mesh_cache[object_id] = mesh_loaded
            else:
                object_mesh_cache[object_id] = None

    obj_mesh = object_mesh_cache.get(object_id)
    if obj_mesh is None:
        return None

    scene_pose_data = _load_scene_pose_json(scene_id, scenes_dir, scene_pose_cache)
    pose_entry = scene_pose_data.get(object_id)
    if pose_entry is None:
        return None

    obj_world_pos = np.array(pose_entry["position"], dtype=np.float32)
    obj_world_quat = np.array(pose_entry["orientation"], dtype=np.float32)
    obj_world_rot_mx = tm.transformations.quaternion_matrix(obj_world_quat)[:3, :3].astype(
        np.float32
    )

    vertices_world = (obj_world_rot_mx @ obj_mesh.vertices.T).T + obj_world_pos
    faces = obj_mesh.faces

    return go.Mesh3d(
        x=vertices_world[:, 0],
        y=vertices_world[:, 1],
        z=vertices_world[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="lightslategray",
        opacity=0.93,
        name=f"Object: {object_id}",
    )


# ---------------------------------------------------------------------------
# Single-figure visualizer  (mirrors visualize_rgbd_dataset.visualize_sample)
# ---------------------------------------------------------------------------


def visualize_group(
    gripper_id: str,
    scene_id: str,
    object_id: str,
    grasp_list: list,  # [(pred_pose, pred_dofs), ...]
    scene_features: torch.Tensor,  # (N, 3)  — raw (unnormalised) coords
    hand_model,
    max_grasps: int = 10,
    seed_pts: Optional[list] = None,  # [[x, y, z], ...] or None
    use_mesh: bool = False,
    object_mesh_trace: Optional[go.Mesh3d] = None,
    randomize_grasps: bool = False,
    rng: Optional[np.random.Generator] = None,
    ground_truth: bool = False,
    dof_mapping: Optional[dict] = None,
    robot_name_mapping: Optional[dict] = None,
    scenes_dir: Optional[str] = None,
    scene_pose_cache: Optional[dict] = None,
) -> go.Figure:
    """
    Build a Plotly figure showing:
      • Scene point cloud  (blue)
    • Up to *max_grasps* predicted gripper poses  (red point clouds or meshes)
      • Grasp-origin markers  (green diamonds)
    • Seed point(s)  (orange stars, if provided)

    Args:
        grasp_list:      List of (pred_pose, pred_dofs, gripper_id, scene_id,
                          object_id) tuples for this group.
        scene_features:  (N, 3) tensor in original world coordinates.
        hand_model:      HandModel instance (from hand_models.pt).
        max_grasps:      Cap on the number of individual grasps to render.
        seed_pts:        Optional list of [x, y, z] seed points to show as orange stars.
        use_mesh:        If True, render each grasp with mesh traces instead of point clouds.
        object_mesh_trace: Optional object mesh trace in world frame.
        randomize_grasps: If True, randomize grasp order before visualization.
        rng:             Optional numpy RNG used when randomize_grasps is enabled.
        ground_truth:    If True, interpret each grasp's pose/dofs as raw
                         ground-truth (object-relative pos+quat_wxyz, native
                         dof order) and convert to world-frame pos+rot6d+
                         dof_mapping order before rendering.
        dof_mapping:     {internal_gripper_name: dof_indices} — required when
                         ground_truth is True.
        robot_name_mapping: {gripper_id: internal_gripper_name} — required
                         when ground_truth is True.
        scenes_dir:      Directory containing <scene_id>/<scene_id>.json scene
                         pose files — required when ground_truth is True.
        scene_pose_cache: Cache dict shared across groups for scene pose JSON.
    """
    fig = go.Figure()

    # ---- Object mesh ----
    if object_mesh_trace is not None:
        fig.add_trace(object_mesh_trace)

    # ---- Seed points (orange stars) ----
    if seed_pts:
        seed_array = np.array(seed_pts, dtype=np.float32)
        if seed_array.ndim == 1:
            seed_array = seed_array.reshape(1, 3)
        fig.add_trace(
            go.Scatter3d(
                x=seed_array[:, 0],
                y=seed_array[:, 1],
                z=seed_array[:, 2],
                mode="markers",
                marker=dict(size=10, color="orange", symbol="diamond", opacity=1.0),
                name="Seed point(s)",
            )
        )

    # ---- Scene point cloud ----
    pts = scene_features.squeeze().cpu()
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="blue", opacity=0.5),
            name="Scene",
        )
    )

    # ---- Predicted grasps ----
    shown = 0
    if randomize_grasps and grasp_list:
        local_rng = rng if rng is not None else np.random.default_rng()
        k = min(max_grasps, len(grasp_list))
        if k < len(grasp_list):
            grasp_indices = local_rng.choice(len(grasp_list), size=k, replace=False).tolist()
        else:
            grasp_indices = local_rng.permutation(len(grasp_list)).tolist()
    else:
        grasp_indices = list(range(len(grasp_list)))

    if randomize_grasps:
        print(
            f"    Randomized grasp indices ({len(grasp_indices)}/{len(grasp_list)}): {grasp_indices}"
        )

    for i, grasp_idx in enumerate(grasp_indices):
        if shown >= max_grasps:
            break

        pose, dofs, grasp_gripper_id, grasp_scene_id, grasp_object_id = grasp_list[grasp_idx]

        if ground_truth:
            cache = scene_pose_cache if scene_pose_cache is not None else {}
            obj_pose = lookup_object_world_pose(grasp_scene_id, grasp_object_id, scenes_dir, cache)
            if obj_pose is None:
                print(
                    f"  [WARN] No scene pose for object '{grasp_object_id}' in scene "
                    f"'{grasp_scene_id}'; skipping ground-truth grasp {i+1}."
                )
                continue
            obj_world_pos, obj_world_quat = obj_pose

            gripper_internal = (robot_name_mapping or {}).get(
                grasp_gripper_id, grasp_gripper_id.lower()
            )
            grasp_dof_indices = (dof_mapping or {}).get(gripper_internal)
            if grasp_dof_indices is None:
                print(
                    f"  [WARN] No dof_mapping entry for gripper '{gripper_internal}'; "
                    f"skipping ground-truth grasp {i+1}."
                )
                continue

            q = reconstruct_q_ground_truth(
                pose, dofs, grasp_dof_indices, obj_world_pos, obj_world_quat
            ).to(device)
        else:
            q = reconstruct_q(pose, dofs).to(device)  # (1, D)

        # Grasp-origin marker
        trans = q[0, :3].cpu()
        fig.add_trace(
            go.Scatter3d(
                x=[trans[0].item()],
                y=[trans[1].item()],
                z=[trans[2].item()],
                mode="markers",
                marker=dict(size=7, color="green", symbol="diamond"),
                name=f"Origin #{i+1}",
                showlegend=(i == 0),
                legendgroup="origins",
            )
        )

        # Robot surface point cloud
        if hand_model is not None:
            try:
                # Determine actual DOF count from the hand model's joint limits
                # (pytorch_kinematics rejects q[:, 9:] if it has more values
                # than the robot's URDF revolute joint count)
                n_actual_dofs = hand_model.revolute_joints_q_lower.shape[1]
                q_trimmed = q[:, : 9 + n_actual_dofs]  # (1, 9+actual)
                q_clean = torch.nan_to_num(q_trimmed, nan=0.0)
                if use_mesh:
                    vis_data = hand_model.get_plotly_data(q=q_clean, opacity=0.5)
                    for j, mesh_trace in enumerate(vis_data):
                        mesh_trace.color = "red"
                        mesh_trace.name = f"Grasp #{i+1}" if j == 0 else None
                        mesh_trace.showlegend = i == 0 and j == 0
                        mesh_trace.legendgroup = "grasps"
                        fig.add_trace(mesh_trace)
                else:
                    robot_pts = (
                        hand_model.get_surface_points(q=q_clean, downsample=True).squeeze().cpu()
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=robot_pts[:, 0],
                            y=robot_pts[:, 1],
                            z=robot_pts[:, 2],
                            mode="markers",
                            marker=dict(size=2, color="red", opacity=0.8),
                            name=f"Grasp #{i+1}",
                            showlegend=(i == 0),
                            legendgroup="grasps",
                        )
                    )
            except Exception as exc:
                print(f"  [WARN] Could not render robot for grasp {i+1}: {exc}")

        shown += 1

    # ---- Layout ----
    fig.update_layout(
        title=f"Gripper: {gripper_id}  |  Scene: {scene_id}  |  Object: {object_id}"
        f"<br><sup>{min(shown, max_grasps)} of {len(grasp_list)} grasps shown</sup>",
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode="cube",
        ),
        width=1100,
        height=850,
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualize predicted grasps from generate_grasps_from_seed_pts.py output."
    )
    parser.add_argument(
        "--grasps",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more *_predicted_grasps.json files to visualize.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.path.join(_REPO_ROOT, "configs"),
        help="Path to Hydra configs directory (default: <repo_root>/configs).",
    )
    parser.add_argument(
        "--indices-json",
        default=None,
        metavar="FILE",
        help="Optional JSON file containing indices to keep (e.g. failures.json). "
        "Supported formats: [1, 4, ...] or {'indices': [1, 4, ...]}.",
    )
    parser.add_argument(
        "--indices-base",
        choices=["auto", "zero", "one"],
        default="auto",
        help="Interpretation of values in --indices-json: auto (default), zero (0-based), one (1-based).",
    )
    parser.add_argument(
        "--max-grasps",
        type=int,
        default=10,
        help="Max number of individual grasps to render per figure (default: 10).",
    )
    parser.add_argument(
        "--filter-gripper",
        default=None,
        help="Only show entries for this gripper ID.",
    )
    parser.add_argument(
        "--filter-scene",
        default=None,
        help="Only show entries for this scene ID.",
    )
    parser.add_argument(
        "--filter-object",
        default=None,
        help="Only show entries for this object ID.",
    )
    parser.add_argument(
        "--seed-pts",
        default=None,
        metavar="FILE",
        help="Optional seed_pts JSON (output of generate_seed_pts.py). When provided, "
        "the seed point for each (gripper, scene, object) is shown as an orange star.",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Render grippers as meshes instead of point clouds (slower, more detailed).",
    )
    parser.add_argument(
        "--object-mesh",
        action="store_true",
        help="Render target object as a transformed mesh (requires objects/ and scenes/ in dataset).",
    )
    parser.add_argument(
        "--object-dir",
        default=None,
        help="Optional object mesh root dir containing GSO/ and YCB/ (default: <dataset_basedir>/objects).",
    )
    parser.add_argument(
        "--scenes-dir",
        default=None,
        help="Optional scene metadata dir containing <scene_id>/<scene_id>.json (default: <dataset_basedir>/scenes).",
    )
    parser.add_argument(
        "--randomize-grasps",
        action="store_true",
        help="Randomize grasp order before selecting up to --max-grasps per group.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional RNG seed used with --randomize-grasps for reproducible selection.",
    )
    parser.add_argument(
        "--group-by",
        choices=["gso", "gs", "scene", "all"],
        default="gs",
        help="How to group grasps into figures: gso=(gripper,scene,object), gs=(gripper,scene), scene=(scene), all=(single figure).",
    )
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Interpret --grasps files as raw ground-truth grasps: pose is "
        "object-relative trans(3)+quat_wxyz(4) and dofs are in native "
        "simulator order. These are "
        "converted to world-frame trans(3)+rot6d(6)+dof_mapping-order "
        "dofs before rendering, matching generated grasps. Requires the "
        "dataset's scenes/ directory (see --scenes-dir) to look up each "
        "object's world pose.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed) if args.randomize_grasps else None

    # ---- Config & paths ----
    cfg = load_cfg(os.path.abspath(args.config_dir))
    dataset_basedir = to_absolute_path(cfg.dataset.dataset_basedir)

    # ---- Hand models ----
    hand_models_path = os.path.join(dataset_basedir, "hand_models.pt")
    print("Loading hand models…")
    if os.path.exists(hand_models_path):
        hand_models: dict = torch.load(hand_models_path, weights_only=False, map_location=device)
    else:
        print(
            f"[WARN] hand_models.pt not found at {hand_models_path} — robot meshes will be skipped."
        )
        hand_models = {}

    # ---- Scene point clouds ----
    scene_pc_path = os.path.join(dataset_basedir, "gnn_scene_adj_point_clouds_new.pt")
    print("Loading scene point clouds…")
    scene_pc_adj: dict = torch.load(scene_pc_path, weights_only=False, map_location="cpu")

    # ---- Gripper internal name mapping (for hand_models lookup) ----
    robot_name_mapping: dict = dict(cfg.dataset.robot_name_mapping)

    # ---- Optional object mesh setup ----
    object_mesh_cache: dict = {}
    scene_pose_cache: dict = {}
    gso_obj_names: set = set()
    object_dir = None
    # scenes_dir is needed both for --object-mesh and for --ground-truth
    # (object-relative -> world-frame pose conversion), so resolve it
    # whenever either is requested.
    scenes_dir = None
    if args.object_mesh or args.ground_truth:
        scenes_dir = (
            os.path.abspath(args.scenes_dir)
            if args.scenes_dir
            else os.path.join(dataset_basedir, "scenes")
        )
    if args.ground_truth and not os.path.isdir(scenes_dir):
        print(
            f"Error: --ground-truth requires scene pose JSONs under '{scenes_dir}' "
            "to convert object-relative poses to world frame, but that directory "
            "does not exist. Pass --scenes-dir to point at the dataset's scenes/ "
            "directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.object_mesh:
        object_dir = (
            os.path.abspath(args.object_dir)
            if args.object_dir
            else os.path.join(dataset_basedir, "objects")
        )
        gso_file = os.path.join(object_dir, "gso_object_ids.txt")
        if os.path.exists(gso_file):
            with open(gso_file) as f:
                gso_obj_names = {line.strip() for line in f if line.strip()}
        else:
            print(
                f"[WARN] gso_object_ids.txt not found at {gso_file}; defaulting unknown IDs to YCB paths."
            )

    # ---- Optional seed points ----
    # seed_pt_lookup[(gripper_id, scene_id, object_id)] = [x, y, z]
    seed_pt_lookup: dict = {}
    if args.seed_pts:
        import json as _json

        with open(args.seed_pts) as f:
            spd = _json.load(f)
        for g_id, scenes in spd.items():
            for s_id, objects in scenes.items():
                for o_id, entry in objects.items():
                    if "seed_pt" in entry:
                        seed_pt_lookup[(g_id, s_id, o_id)] = entry["seed_pt"]

    # Build per-group seed-point list from the global seed lookup.
    def _collect_seed_points_for_group(gripper_id: str, scene_id: str, object_id: str) -> list:
        if not seed_pt_lookup:
            return []

        if args.group_by == "gso":
            pt = seed_pt_lookup.get((gripper_id, scene_id, object_id))
            return [pt] if pt is not None else []

        if args.group_by == "gs":
            return [
                p for (g, s, _o), p in seed_pt_lookup.items() if g == gripper_id and s == scene_id
            ]

        if args.group_by == "scene":
            return [p for (_g, s, _o), p in seed_pt_lookup.items() if s == scene_id]

        # all
        return list(seed_pt_lookup.values())

    # ---- Optional global index filter ----
    indices_filter_values = None
    if args.indices_json:
        indices_filter_values = load_indices_json(args.indices_json)
        print(f"Loaded {len(indices_filter_values)} indices from {args.indices_json}")

    # ---- Process each input file ----
    total_figs = 0
    for grasp_file in args.grasps:
        print(f"\nLoading: {grasp_file}")
        data = load_json_grasps(grasp_file)
        if indices_filter_values is not None:
            data = filter_grasp_data_by_indices(
                data=data,
                index_values=indices_filter_values,
                index_base=args.indices_base,
                source_name=grasp_file,
            )
        groups = group_grasps(data, group_by=args.group_by)
        group_sizes = [len(v) for v in groups.values()]
        if group_sizes:
            print(
                f"  Grouping mode={args.group_by}: {len(group_sizes)} groups "
                f"(min={min(group_sizes)}, max={max(group_sizes)}, mean={sum(group_sizes)/len(group_sizes):.2f})"
            )
            if args.randomize_grasps and max(group_sizes) <= 1:
                print(
                    "  [INFO] All groups have 1 grasp; use --group-by gs/scene/all for visible randomization."
                )

        for group_key, grasp_list in sorted(groups.items()):
            gripper_id, scene_id, object_id = _expand_group_key(group_key)

            # Optional filters
            if args.filter_gripper and gripper_id != args.filter_gripper:
                continue
            if args.filter_scene and scene_id != args.filter_scene:
                continue
            if args.filter_object and object_id != args.filter_object:
                continue

            print(
                f"  Visualizing: {gripper_id} | {scene_id} | {object_id}  ({len(grasp_list)} grasps)"
            )

            # ---- Scene features ----
            if scene_id not in scene_pc_adj:
                print(f"  [WARN] Scene '{scene_id}' not in scene_pc_adj, skipping.")
                continue
            scene_data = scene_pc_adj[scene_id]
            scene_features = scene_data["scene_features"]  # (N,3) or (1,N,3)
            if scene_features.dim() == 3:
                scene_features = scene_features.squeeze(0)  # → (N, 3)

            # ---- Hand model (use internal name for lookup) ----
            gripper_internal = robot_name_mapping.get(gripper_id, gripper_id.lower())
            hand_model = hand_models.get(gripper_internal)
            if hand_model is None:
                print(f"  [WARN] No hand model found for '{gripper_internal}', robot mesh skipped.")
            else:
                hand_model = hand_model.to(device)

            # ---- Build & show figure ----
            object_mesh_trace = None
            if args.object_mesh and object_dir is not None and scenes_dir is not None:
                object_mesh_trace = _build_object_mesh_trace(
                    scene_id=scene_id,
                    object_id=object_id,
                    object_dir=object_dir,
                    scenes_dir=scenes_dir,
                    gso_obj_names=gso_obj_names,
                    object_mesh_cache=object_mesh_cache,
                    scene_pose_cache=scene_pose_cache,
                )
                if object_mesh_trace is None:
                    print(
                        f"  [WARN] Could not build object mesh for '{object_id}' in scene '{scene_id}'."
                    )

            fig = visualize_group(
                gripper_id=gripper_id,
                scene_id=scene_id,
                object_id=object_id,
                grasp_list=grasp_list,
                scene_features=scene_features,
                hand_model=hand_model,
                max_grasps=args.max_grasps,
                seed_pts=_collect_seed_points_for_group(gripper_id, scene_id, object_id),
                use_mesh=args.mesh,
                object_mesh_trace=object_mesh_trace,
                randomize_grasps=args.randomize_grasps,
                rng=rng,
                ground_truth=args.ground_truth,
                dof_mapping=dict(cfg.dataset.dof_mapping),
                robot_name_mapping=robot_name_mapping,
                scenes_dir=scenes_dir,
                scene_pose_cache=scene_pose_cache,
            )
            fig.show()
            total_figs += 1

    print(f"\nDone. Showed {total_figs} figure(s).")


if __name__ == "__main__":
    main()
