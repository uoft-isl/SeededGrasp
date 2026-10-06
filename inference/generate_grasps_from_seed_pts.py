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
generate_grasps_from_seed_pts.py

Generate robot grasps from VLM-provided seed points using the SeededGraspFlow model.

Usage (run from the repo root so that Hydra can find configs):

    python inference/generate_grasps_from_seed_pts.py \
        --seed-pts vlm_output/seed_pts_test.json \
        --output-dir generated_grasps/ \
        [--num-samples 10] \
        [--time-steps 10] \
        [--cfg-weight 1.0] \
        [--plot-grasps]

For each seed-point entry, the script runs generation from the configured
initial orientation variants. Prompt metadata is preserved in the exported
grasp JSON so each predicted grasp can be traced back to the prompt that
produced its seed point.

Input JSON format (legacy seed_pts_test.json):
    {
        "<gripper_id>": {
            "<scene_id>": {
                "<object_id>": {
                    "seed_pt": [x, y, z],   # 3-D world-frame seed point
                    ...
                }
            }
        }
    }

Input JSON format (prompt-aware seed points):
    {
        "<gripper_id>": {
            "<scene_id>": {
                "<object_id>": {
                    "<prompt_source>": {
                        "<prompt_key>": {
                            "seed_pt": [x, y, z],
                            "prompt": "...",
                            "prompt_index": 1,
                            "prompt_source": "easy",
                            "obj_name_simple": "...",
                            ...
                        }
                    }
                }
            }
        }
    }

Output: one JSON file per gripper written to <output_dir>/<gripper_id>_predicted_grasps.json

Output JSON format matches generate_grasps_flow.py's export schema:
    {
        "scene_id":   ["<scene_id>",  ...],   # one entry per generated grasp
        "object_id":  ["<object_id>", ...],
        "gripper_id": ["<gripper_id>", ...],
        "pred_pose":  [[trans(3) + rot6d(6)], ...],
        "pred_dofs":  [[dofs], ...],
        "seed_pt":    [[x, y, z], ...],
        "init_z_rot_deg": [0 | 90 | 180, ...],
        "prompt_source": ["easy" | "difficult" | ...],
        "prompt_key": ["prompt_1" | ...],
        "prompt_index": [1 | ...],
        "prompt": ["...", ...],
        "obj_name_simple": ["...", ...]
    }
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# ---------------------------------------------------------------------------
# Make sure the repo root is on the path when called as a script
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pypose as pp

import models.flow
from utils import math_utils
from utils.normalization import normalize_pc, normalize_q, unnormalize_q
from utils_data.augmentors import sample_pc_noise_random
from utils_data.rgbd_scene_dataset import (
    apply_gripper_rotation_alignment_with_transforms,
    build_gripper_alignment_transforms,
)

# mixed_eulers_method is defined locally below with CFG scheduling support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INIT_Z_ROT_DEGS = [0]


# ---------------------------------------------------------------------------
# ODE integrators
# ---------------------------------------------------------------------------


def mixed_eulers_method(x, time_steps, model, params, w=1.0):
    """Euler integration over the mixed SE(3) × R^N state space.

    Args:
        w: CFG weight. Either a float or a callable w(t) -> float.
    """
    with torch.no_grad():
        delta_t = time_steps[1] - time_steps[0]
        x_trans = x[:, :3]
        x_rot_9d = math_utils.robust_compute_rotation_matrix_from_ortho6d(x[:, 3:9])
        x_joints = x[:, 9:]

        all_x = [x]
        for t_i in range(len(time_steps)):
            t = time_steps[t_i]
            u_theta = model(
                x, torch.tensor([t], dtype=torch.float32).to(device).unsqueeze(0), *params
            )

            w_t = w(t.item()) if callable(w) else w
            if w_t != 1.0:
                # Unconditional branch must match training's CFG-dropout mechanism
                # exactly: pass drop_all as the model's own drop_mask argument so
                # it zeroes obj_pc/obj_adj/seed_pt AND robot_pc/robot_adj/
                # query_tensor together (see models.flow.SeededGraspFlow.forward),
                # not a hand-rolled partial zeroing (see generate_grasps_flow.py's
                # mixed_eulers_method for the reference implementation).
                drop_all = torch.ones(x.shape[0], dtype=torch.bool, device=device)
                u_theta_uncond = model(
                    x,
                    torch.tensor([t], dtype=torch.float32).to(device).unsqueeze(0),
                    *params,
                    drop_all,
                )
                u_theta = w_t * u_theta + (1.0 - w_t) * u_theta_uncond

            u_theta_trans = u_theta[:, :3]
            u_theta_rot = u_theta[:, 3:6]
            u_theta_so3 = pp.so3(u_theta_rot)
            u_theta_joints = u_theta[:, 6:]

            x_trans = x_trans + delta_t * u_theta_trans
            x_rot_9d = x_rot_9d @ math_utils.matrix_from_quat(pp.so3(delta_t * u_theta_so3).Exp())
            x_joints = x_joints + delta_t * u_theta_joints

            x_rot_6d = x_rot_9d[:, :, :2].transpose(1, 2).reshape(x.shape[0], -1)
            x = torch.cat([x_trans, x_rot_6d, x_joints], dim=1)

            all_x.append(x)
        all_x.append(x)
        return all_x


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def load_cfg(config_dir: str, config_name: str = "config_rgbd_scene") -> DictConfig:
    """Load a Hydra config from config_dir without entering @hydra.main."""
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def build_model(cfg: DictConfig, dataset_basedir: str) -> torch.nn.Module:
    """Instantiate SeededGraspFlow and load the latest weights."""
    robot_name_list = [cfg.dataset.robot_name_mapping[r] for r in cfg.dataset.robot_name_list]
    robot_models_path = os.path.join(dataset_basedir, "hand_models.pt")
    robot_models = torch.load(robot_models_path, weights_only=False, map_location=device)

    # Drop robots that are not in our name list
    for k in list(robot_models.keys()):
        if k not in robot_name_list:
            del robot_models[k]

    model = models.flow.SeededGraspFlow(
        cfg=cfg, robot_models=robot_models, robot_models_path=robot_models_path
    ).to(device)
    model.eval()
    return model


def find_model_weights(cfg: DictConfig, model_path: str = None) -> str:
    """Return the checkpoint to load: an explicit override, or the latest
    experiment directory matching cfg.train.exp_name under cfg.train.log_basedir."""
    if model_path:
        return model_path
    log_basedir = to_absolute_path(cfg.train.log_basedir)
    # Find the most recently modified log directory (NOT a string sort on the
    # directory name: train_flow_ddp.py names runs "{exp_name}_ddp_{timestamp}"
    # while train_flow.py uses "{exp_name}_{timestamp}", and "ddp" sorts after
    # any digit, so a string sort would always prefer a DDP run over a newer
    # non-DDP one regardless of actual date).
    log_dirs = [d for d in os.listdir(log_basedir) if d.startswith(cfg.train.exp_name + "_")]
    if not log_dirs:
        raise FileNotFoundError(
            f"No log directories found for experiment {cfg.train.exp_name} in {log_basedir}"
        )
    latest_log_dir = max(log_dirs, key=lambda d: os.path.getmtime(os.path.join(log_basedir, d)))
    return os.path.join(log_basedir, latest_log_dir, cfg.train.weights_dir, cfg.train.weights_name)


def get_legacy_base_init_rot6d(robot_name: str) -> torch.Tensor:
    """Return the base 6D orientation used for each robot's initial pose."""
    robot_lower = robot_name.lower()
    if robot_lower == "franka_panda":
        return torch.tensor([-1, 0, 0, 0, 1, 0], dtype=torch.float32, device=device)
    if robot_lower == "allegro":
        return torch.tensor([0, 0, -1, -1, 0, 0], dtype=torch.float32, device=device)
    if robot_lower == "robotiq_3finger":
        return torch.tensor([-1, 0, 0, 0, 0, -1], dtype=torch.float32, device=device)
    raise Exception("Robot name not found.")


def rotate_rot6d_about_z(rot6d: torch.Tensor, z_rot_deg: float) -> torch.Tensor:
    """Left-compose a 6D rotation with a world-frame z-axis rotation."""
    rot_mat = math_utils.robust_compute_rotation_matrix_from_ortho6d(rot6d.unsqueeze(0)).squeeze(0)
    angle_rad = np.deg2rad(z_rot_deg)
    cos_z = float(np.cos(angle_rad))
    sin_z = float(np.sin(angle_rad))
    rot_z = torch.tensor(
        [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=rot6d.device,
    )
    rotated = rot_z @ rot_mat
    return rotated[:, :2].transpose(0, 1).reshape(-1)


def plot_generated_grasps(
    gripper_id: str,
    scene_id: str,
    object_id: str,
    seed_pt: torch.Tensor,
    scene_features: torch.Tensor,
    hand_model,
    grasps_by_rot: dict,
    init_grasps_by_rot: dict = None,
):
    """Overlay all generated grasps for different z-rotation initializations.

    Args:
        init_grasps_by_rot: Dict mapping z_rot_deg -> list of initial pose tensors
            (shape (1, 9+dof_dim)). When provided, the initial hand poses are drawn
            alongside the final generated poses using open markers.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.renderers.default = "browser"

    variant_colors = {0: "red", 90: "green", 180: "orange"}
    fig = go.Figure()

    pts = scene_features.squeeze().detach().cpu()
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="blue", opacity=0.35),
            name="Scene",
        )
    )

    seed_xyz = seed_pt.squeeze().detach().cpu()
    fig.add_trace(
        go.Scatter3d(
            x=[seed_xyz[0].item()],
            y=[seed_xyz[1].item()],
            z=[seed_xyz[2].item()],
            mode="markers",
            marker=dict(size=10, color="black", symbol="diamond"),
            name="Seed point",
        )
    )

    # ---- Initial poses ----
    if init_grasps_by_rot:
        for z_rot_deg in INIT_Z_ROT_DEGS:
            init_q_list = init_grasps_by_rot.get(z_rot_deg, [])
            color = variant_colors[z_rot_deg]
            for grasp_idx, grasp_q in enumerate(init_q_list):
                trans = grasp_q[0, :3].detach().cpu()
                fig.add_trace(
                    go.Scatter3d(
                        x=[trans[0].item()],
                        y=[trans[1].item()],
                        z=[trans[2].item()],
                        mode="markers",
                        marker=dict(size=7, color=color, symbol="circle-open", line=dict(width=2)),
                        name=f"init pose z={z_rot_deg} deg",
                        legendgroup=f"init_z_rot_{z_rot_deg}",
                        showlegend=(grasp_idx == 0),
                    )
                )

                if hand_model is not None:
                    try:
                        robot_pts = (
                            hand_model.get_surface_points(q=grasp_q.to(device), downsample=True)
                            .squeeze()
                            .detach()
                            .cpu()
                        )
                        fig.add_trace(
                            go.Scatter3d(
                                x=robot_pts[:, 0],
                                y=robot_pts[:, 1],
                                z=robot_pts[:, 2],
                                mode="markers",
                                marker=dict(size=2, color=color, opacity=0.25),
                                name=f"init pose z={z_rot_deg} deg hand",
                                legendgroup=f"init_z_rot_{z_rot_deg}",
                                showlegend=False,
                            )
                        )
                    except Exception as exc:
                        print(
                            f"  [WARN] Could not plot init hand for z={z_rot_deg} deg grasp {grasp_idx + 1}: {exc}"
                        )

    # ---- Final generated poses ----
    for z_rot_deg in INIT_Z_ROT_DEGS:
        grasp_q_list = grasps_by_rot.get(z_rot_deg, [])
        color = variant_colors[z_rot_deg]
        for grasp_idx, grasp_q in enumerate(grasp_q_list):
            trans = grasp_q[0, :3].detach().cpu()
            fig.add_trace(
                go.Scatter3d(
                    x=[trans[0].item()],
                    y=[trans[1].item()],
                    z=[trans[2].item()],
                    mode="markers",
                    marker=dict(size=6, color=color, symbol="diamond"),
                    name=f"final z={z_rot_deg} deg",
                    legendgroup=f"z_rot_{z_rot_deg}",
                    showlegend=(grasp_idx == 0),
                )
            )

            if hand_model is not None:
                try:
                    robot_pts = (
                        hand_model.get_surface_points(q=grasp_q.to(device), downsample=True)
                        .squeeze()
                        .detach()
                        .cpu()
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=robot_pts[:, 0],
                            y=robot_pts[:, 1],
                            z=robot_pts[:, 2],
                            mode="markers",
                            marker=dict(size=2, color=color, opacity=0.6),
                            name=f"final z={z_rot_deg} deg hand",
                            legendgroup=f"z_rot_{z_rot_deg}",
                            showlegend=False,
                        )
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] Could not plot hand for z={z_rot_deg} deg grasp {grasp_idx + 1}: {exc}"
                    )

    fig.update_layout(
        title=(
            f"{gripper_id} | {scene_id} | {object_id}"
            f"<br><sup>Initial z-rotation variants: {INIT_Z_ROT_DEGS} — open markers = initial pose, filled = final pose</sup>"
        ),
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
    fig.show()


# ---------------------------------------------------------------------------
# Per-sample initialisation (mirrors generate_grasps_flow.py lines 420-425)
# ---------------------------------------------------------------------------


# Leading joint values for the canonical initial pose, per gripper. Anything
# beyond these (and any gripper not listed) stays at zero.
_INITIAL_DOFS = {
    "allegro": [0.0] * 12 + [0.75, 0.0, 0.25, 0.0],  # thumb pre-posed
    "franka_panda": [0.04, 0.04],  # fingers open
}


def get_initial_x(
    robot_name: str,
    init_rot6d: torch.Tensor,
    seed_pt: torch.tensor,
    pad_q: int,
    z_rot_deg: float = 0.0,
) -> torch.Tensor:
    """Return a sensible initial pose tensor padded to pad_q dims.

    The model expects (batch, pad_q) where pad_q = 9 pose dims + the padded
    joint width. Unused DOF slots are zero (NOT NaN) -- NaN would propagate
    through normalize_q. This matches generate_grasps_flow.py, which uses
    torch.zeros_like for padding.
    """
    init_trans = seed_pt[0, 0, :].clone()
    init_trans[2] += 0.05
    rot6d = rotate_rot6d_about_z(init_rot6d, z_rot_deg=z_rot_deg)
    pose = torch.cat([init_trans, rot6d], dim=0)

    num_dof_slots = pad_q - 9
    dofs = torch.zeros(num_dof_slots, dtype=torch.float32, device=device)
    initial = _INITIAL_DOFS.get(robot_name.lower())
    if initial is not None:
        n = min(len(initial), num_dof_slots)
        dofs[:n] = torch.tensor(initial[:n], dtype=torch.float32, device=device)
    return torch.cat([pose, dofs], dim=0).unsqueeze(0)


def iter_seed_point_entries(seed_pts_data: dict):
    """Yield flattened seed-point entries from legacy or prompt-aware JSON."""
    for gripper_id, scenes in seed_pts_data.items():
        for scene_id, objects in scenes.items():
            for object_id, object_entry in objects.items():
                if isinstance(object_entry, dict) and "seed_pt" in object_entry:
                    yield {
                        "gripper_id": gripper_id,
                        "scene_id": scene_id,
                        "object_id": object_id,
                        "prompt_source": object_entry.get("prompt_source", "unknown"),
                        "prompt_key": object_entry.get("prompt_key", "seed_pt"),
                        "prompt_index": object_entry.get("prompt_index", 1),
                        "prompt": object_entry.get("prompt", ""),
                        "obj_name_simple": object_entry.get("obj_name_simple"),
                        "seed_pt": object_entry["seed_pt"],
                    }
                    continue

                if not isinstance(object_entry, dict):
                    continue

                for prompt_source, prompt_entries in object_entry.items():
                    if not isinstance(prompt_entries, dict):
                        continue
                    for prompt_key, prompt_entry in prompt_entries.items():
                        if not isinstance(prompt_entry, dict) or "seed_pt" not in prompt_entry:
                            continue
                        yield {
                            "gripper_id": gripper_id,
                            "scene_id": scene_id,
                            "object_id": object_id,
                            "prompt_source": prompt_entry.get("prompt_source", prompt_source),
                            "prompt_key": prompt_key,
                            "prompt_index": prompt_entry.get("prompt_index", 1),
                            "prompt": prompt_entry.get("prompt", ""),
                            "obj_name_simple": prompt_entry.get("obj_name_simple"),
                            "seed_pt": prompt_entry["seed_pt"],
                        }


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------


def generate(
    seed_pts_path: str,
    output_dir: str,
    cfg_path: str,
    num_samples: int = 10,
    num_time_steps: int = 5,
    cfg_weight: float = None,
    grippers: list = None,
    plot_grasps: bool = False,
    model_path: str = None,
):
    """Run grasp generation for all entries in seed_pts_path.

    Args:
        grippers: Optional list of gripper IDs to process (e.g. ['Allegro']).
                  If None or empty, all grippers in the seed_pts file are processed.
    """

    # ---- Config ----
    cfg = load_cfg(config_dir=os.path.abspath(cfg_path))
    dataset_basedir = to_absolute_path(cfg.dataset.dataset_basedir)
    print(dataset_basedir)

    gripper_alignment_rot6d = getattr(cfg.dataset, "gripper_alignment_rot6d", None)
    canonical_rot6d_cfg = getattr(cfg.dataset, "canonical_rot6d", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    use_rotation_alignment = bool(gripper_alignment_rot6d)
    alignment_transforms = build_gripper_alignment_transforms(
        gripper_alignment_rot6d=gripper_alignment_rot6d,
        canonical_rot6d=canonical_rot6d_cfg,
    )
    warned_unknown_grippers = set()

    # ---- Model ----
    print("Loading model…")
    model = build_model(cfg, dataset_basedir)
    weight_path = find_model_weights(cfg, model_path=model_path)
    print(f"Loading weights from: {weight_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weights not found at {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # ---- Scene point clouds ----
    print("Loading scene point clouds…")
    scene_pc_adj = torch.load(
        os.path.join(dataset_basedir, "gnn_scene_adj_point_clouds_new.pt"),
        weights_only=False,
        map_location="cpu",
    )

    hand_models = None
    if plot_grasps:
        print("Loading hand models for plotting…")
        hand_models = torch.load(
            os.path.join(dataset_basedir, "hand_models.pt"),
            weights_only=False,
            map_location=device,
        )

    # DOF dimensionality comes from dof_mapping (same source as flow.py:1043:
    #   dof_len = len(self.cfg.dataset.dof_mapping[rname]))
    dof_dims: dict = {
        rname: len(cfg.dataset.dof_mapping[rname]) for rname in cfg.dataset.dof_mapping
    }

    # Total padded q length expected by the model: 9 pose dims + padded joints.
    PAD_Q = cfg.dataset.pad_q

    # Fall back to the config's guidance weight when --cfg-weight is omitted.
    if cfg_weight is None:
        cfg_weight = cfg.model.w
    print(f"Classifier-free guidance w = {cfg_weight}, {num_time_steps} Euler steps")

    # ---- Seed points ----
    with open(seed_pts_path, "r") as f:
        seed_pts_data: dict = json.load(f)

    # ---- Time steps ----
    delta_t = 1.0 / num_time_steps
    time_steps = torch.tensor(
        np.linspace(0, 1.0 - delta_t, num_time_steps), dtype=torch.float32
    ).to(device)

    seed_entries = list(iter_seed_point_entries(seed_pts_data))
    total = len(seed_entries)
    print(f"Generating grasps for {total} seed-point entries…\n")

    processed = 0
    os.makedirs(output_dir, exist_ok=True)

    # Apply gripper filter
    gripper_ids_to_run = [g for g in seed_pts_data if not grippers or g in grippers]
    if grippers:
        skipped = set(seed_pts_data) - set(gripper_ids_to_run)
        if skipped:
            print(f"Skipping grippers not in filter: {sorted(skipped)}")

    for gripper_id in gripper_ids_to_run:
        scenes = seed_pts_data[gripper_id]
        gripper_internal = cfg.dataset.robot_name_mapping.get(gripper_id, gripper_id.lower())
        dof_dim = dof_dims.get(gripper_internal, PAD_Q - 9)
        if use_rotation_alignment:
            init_rot6d = torch.tensor(canonical_rot6d_cfg, dtype=torch.float32, device=device)
            if init_rot6d.numel() != 6:
                raise ValueError("cfg.dataset.canonical_rot6d must contain exactly 6 values")
        else:
            init_rot6d = get_legacy_base_init_rot6d(gripper_internal)

        # Accumulate all results for this gripper (mirrors generate_grasps_flow.py)
        grasps_export = {
            "scene_id": [],
            "object_id": [],
            "gripper_id": [],
            "pred_pose": [],
            "pred_dofs": [],
            "seed_pt": [],
            "init_z_rot_deg": [],
            "prompt_source": [],
            "prompt_key": [],
            "prompt_index": [],
            "prompt": [],
            "obj_name_simple": [],
        }

        hand_model = None
        if plot_grasps:
            hand_model = hand_models.get(gripper_internal)
            if hand_model is not None:
                hand_model = hand_model.to(device)
            else:
                print(
                    f"  [WARN] No hand model found for '{gripper_internal}', plot will show origins only."
                )

        for scene_id, objects in scenes.items():
            # ---- Load scene features ----
            if scene_id not in scene_pc_adj:
                print(f"  [WARN] Scene '{scene_id}' not found in scene_pc_adj, skipping.")
                continue
            scene_data = scene_pc_adj[scene_id]
            scene_adj_raw = scene_data["scene_adj"]
            scene_features = scene_data["scene_features"]  # (N, 3) or (1, N, 3)

            # Ensure (1, N, 3)
            if scene_features.dim() == 2:
                scene_features = scene_features.unsqueeze(0)
            scene_features = scene_features.to(device)

            # Convert sparse adj to dense
            if scene_adj_raw.is_sparse:
                scene_adj = scene_adj_raw.to_dense().unsqueeze(0).to(device)
            else:
                scene_adj = (
                    scene_adj_raw.unsqueeze(0).to(device)
                    if scene_adj_raw.dim() == 2
                    else scene_adj_raw.to(device)
                )

            for object_id, entry in objects.items():
                prompt_entries = []
                if isinstance(entry, dict) and "seed_pt" in entry:
                    prompt_entries = [
                        {
                            "prompt_source": entry.get("prompt_source", "unknown"),
                            "prompt_key": entry.get("prompt_key", "seed_pt"),
                            "prompt_index": entry.get("prompt_index", 1),
                            "prompt": entry.get("prompt", ""),
                            "obj_name_simple": entry.get("obj_name_simple"),
                            "seed_pt": entry["seed_pt"],
                        }
                    ]
                else:
                    for prompt_source, prompt_dict in entry.items():
                        if not isinstance(prompt_dict, dict):
                            continue
                        for prompt_key, prompt_entry in prompt_dict.items():
                            if isinstance(prompt_entry, dict) and "seed_pt" in prompt_entry:
                                prompt_entries.append(
                                    {
                                        "prompt_source": prompt_entry.get(
                                            "prompt_source", prompt_source
                                        ),
                                        "prompt_key": prompt_key,
                                        "prompt_index": prompt_entry.get("prompt_index", 1),
                                        "prompt": prompt_entry.get("prompt", ""),
                                        "obj_name_simple": prompt_entry.get("obj_name_simple"),
                                        "seed_pt": prompt_entry["seed_pt"],
                                    }
                                )

                for prompt_entry in prompt_entries:
                    processed += 1
                    seed_pt_raw = torch.tensor(prompt_entry["seed_pt"], dtype=torch.float32)  # (3,)
                    # Shape expected by model: (B, 1, 3)
                    seed_pt = seed_pt_raw.unsqueeze(0).unsqueeze(0).to(device)
                    grasps_by_rot = {z_rot_deg: [] for z_rot_deg in INIT_Z_ROT_DEGS}
                    init_grasps_by_rot = {z_rot_deg: [] for z_rot_deg in INIT_Z_ROT_DEGS}

                    prompt_source = prompt_entry["prompt_source"]
                    prompt_key = prompt_entry["prompt_key"]
                    prompt_index = prompt_entry["prompt_index"]
                    prompt_text = prompt_entry["prompt"]
                    obj_name_simple = prompt_entry.get("obj_name_simple")

                    print(
                        f"  [{processed}/{total}] {gripper_id} | {scene_id} | {object_id} | {prompt_source} | {prompt_key}"
                    )

                    # ---- Augmentation ----
                    if cfg.dataset.augment_pc:
                        random_pc_noise = sample_pc_noise_random(
                            1, scene_features.shape[1], device=device
                        )
                        scene_features_aug = scene_features + random_pc_noise
                    else:
                        scene_features_aug = scene_features

                    # ---- Normalization ----
                    if cfg.dataset.normalize_q:
                        obj_features_norm = normalize_pc(cfg, scene_features_aug)
                        seed_pt_norm = normalize_pc(cfg, seed_pt)
                    else:
                        obj_features_norm = scene_features_aug
                        seed_pt_norm = seed_pt

                    robot_name_idx = torch.tensor(
                        [model.robot_name_to_idx[gripper_internal]], device=device
                    )
                    params = (obj_features_norm, scene_adj, robot_name_idx, seed_pt_norm)

                    # ---- Run num_samples independent samples for each init rotation ----
                    for z_rot_deg in INIT_Z_ROT_DEGS:
                        for _ in range(num_samples):
                            x = get_initial_x(
                                gripper_internal,
                                init_rot6d,
                                seed_pt,
                                pad_q=PAD_Q,
                                z_rot_deg=z_rot_deg,
                            )
                            init_q_for_plot = x[:, : 9 + dof_dim]
                            if use_rotation_alignment:
                                init_q_for_plot = apply_gripper_rotation_alignment_with_transforms(
                                    q=init_q_for_plot,
                                    gripper_id=gripper_internal,
                                    transforms=alignment_transforms,
                                    inverse=True,
                                    warned_unknown_grippers=warned_unknown_grippers,
                                )
                            init_grasps_by_rot[z_rot_deg].append(init_q_for_plot.detach().cpu())
                            x = normalize_q(cfg, x, [gripper_internal])
                            # Zero out padding DOF slots after normalization.
                            # The reference (generate_grasps_flow.py:486) does x[mask]=0.0
                            # using the NaN mask from the dataset's q, which has NaN in the
                            # padding positions.  Without this the xd_embedder sees -1.0
                            # (the normalised value of 0) in every padding slot instead of 0.0.
                            x[:, 9 + dof_dim :] = 0.0

                            trajectory = mixed_eulers_method(
                                x, time_steps, model, params, w=cfg_weight
                            )
                            final_x = trajectory[-1]

                            if cfg.dataset.normalize_q:
                                final_x = unnormalize_q(cfg, final_x, [gripper_internal])

                            if use_rotation_alignment:
                                final_x = apply_gripper_rotation_alignment_with_transforms(
                                    q=final_x,
                                    gripper_id=gripper_internal,
                                    transforms=alignment_transforms,
                                    inverse=True,
                                    warned_unknown_grippers=warned_unknown_grippers,
                                )

                            final_q = final_x[:, : 9 + dof_dim].detach().cpu()
                            final_np = final_q.squeeze(0).numpy()
                            grasps_by_rot[z_rot_deg].append(final_q)

                            grasps_export["scene_id"].append(scene_id)
                            grasps_export["object_id"].append([object_id])
                            grasps_export["gripper_id"].append([gripper_id])
                            grasps_export["pred_pose"].append(final_np[:9].tolist())
                            grasps_export["pred_dofs"].append(final_np[9 : 9 + dof_dim].tolist())
                            grasps_export["seed_pt"].append(seed_pt_raw.cpu().tolist())
                            grasps_export["init_z_rot_deg"].append(z_rot_deg)
                            grasps_export["prompt_source"].append(prompt_source)
                            grasps_export["prompt_key"].append(prompt_key)
                            grasps_export["prompt_index"].append(prompt_index)
                            grasps_export["prompt"].append(prompt_text)
                            grasps_export["obj_name_simple"].append(obj_name_simple)

                    print(f"    {num_samples * len(INIT_Z_ROT_DEGS)} grasps generated")

                    if plot_grasps:
                        plot_generated_grasps(
                            gripper_id=gripper_id,
                            scene_id=scene_id,
                            object_id=object_id,
                            seed_pt=seed_pt,
                            scene_features=scene_features,
                            hand_model=hand_model,
                            grasps_by_rot=grasps_by_rot,
                            init_grasps_by_rot=init_grasps_by_rot,
                        )

        # ---- Write one file per gripper ----
        out_path = os.path.join(output_dir, f"{gripper_id}_predicted_grasps.json")
        with open(out_path, "w") as f:
            json.dump(grasps_export, f, indent=4)
        n = len(grasps_export["pred_pose"])
        print(f"  → {out_path} ({n} total grasps)\n")

    print(f"Done. Generated grasps for {processed} (scene, object) entries.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate robot grasps from VLM seed points using SeededGraspFlow."
    )
    parser.add_argument(
        "--seed-pts",
        required=True,
        help="Path to seed points JSON (e.g. vlm_output/seed_pts_test.json).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write per-gripper output JSON files.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.path.join(_REPO_ROOT, "configs"),
        help="Path to the Hydra configs directory (default: <repo_root>/configs).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of grasp samples to generate per (gripper, scene, object) (default: 1).",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=5,
        help="Number of Euler integration steps (default: 5, i.e. step size 0.2).",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=None,
        help="Classifier-free guidance weight w. Defaults to model.w in the "
        "Hydra config (1.1); w = 1.0 disables guidance.",
    )
    parser.add_argument(
        "--grippers",
        nargs="+",
        default=None,
        metavar="GRIPPER",
        help="Only generate grasps for these gripper IDs (e.g. --grippers Allegro franka_panda). "
        "Default: all grippers in the seed-pts file.",
    )
    parser.add_argument(
        "--plot-grasps",
        action="store_true",
        default=False,
        help="Overlay grasps from init z-rotations 0/90/180 deg in one Plotly figure per (gripper, scene, object).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Explicit path to a model checkpoint. Default: latest experiment directory "
        "matching cfg.train.exp_name under cfg.train.log_basedir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        seed_pts_path=args.seed_pts,
        output_dir=args.output_dir,
        cfg_path=args.config_dir,
        num_samples=args.num_samples,
        num_time_steps=args.time_steps,
        cfg_weight=args.cfg_weight,
        grippers=args.grippers,
        plot_grasps=args.plot_grasps,
        model_path=args.model_path,
    )
