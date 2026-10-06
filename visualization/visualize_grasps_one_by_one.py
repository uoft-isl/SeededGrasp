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
visualize_grasps_one_by_one.py

Visualize predicted grasps one at a time with the intended prompt.

Usage (run from the repo root):

    python visualization/visualize_grasps_one_by_one.py \
        --grasps ./Allegro_predicted_grasps.json \
        [--config-dir configs] \
        [--max-items 50] \
        [--filter-gripper Allegro] \
        [--filter-scene scene-multiple-0ecd8c536ec6b29f] \
        [--filter-object AllergenFree_JarroDophilus] \
        [--filter-prompt-substring "white cap"] \
        [--mesh] \
        [--no-pause]

Input JSON format:
- Supports output from scripts/generate_grasps_from_seed_pts.py, including prompt fields.
- Also supports older grasp files that do not contain prompt metadata.
"""

import argparse
import json
import os
import sys
from typing import Optional

import plotly.graph_objects as go
import plotly.io as pio
import torch

# ---------------------------------------------------------------------------
# Repo root on sys.path so project imports work
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path

pio.renderers.default = "browser"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_cfg(config_dir: str, config_name: str = "config_rgbd_scene"):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _as_scalar_id(value):
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return value[0]
    return value


def _safe_get_list_value(data: dict, key: str, idx: int, default=None):
    values = data.get(key)
    if isinstance(values, list) and idx < len(values):
        return values[idx]
    return default


def reconstruct_q(pred_pose: list, pred_dofs: list) -> torch.Tensor:
    pose = torch.tensor(pred_pose, dtype=torch.float32)
    dofs = torch.tensor(pred_dofs, dtype=torch.float32)
    return torch.cat([pose, dofs], dim=0).unsqueeze(0)


def build_figure(
    gripper_id: str,
    scene_id: str,
    object_id: str,
    prompt_text: str,
    prompt_source: str,
    prompt_key: str,
    seed_pt: Optional[list],
    pred_pose: list,
    pred_dofs: list,
    scene_features: torch.Tensor,
    hand_model,
    use_mesh: bool,
    item_idx: int,
    total_items: int,
) -> go.Figure:
    fig = go.Figure()

    # Scene
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

    # Seed point
    if seed_pt is not None and len(seed_pt) == 3:
        fig.add_trace(
            go.Scatter3d(
                x=[seed_pt[0]],
                y=[seed_pt[1]],
                z=[seed_pt[2]],
                mode="markers",
                marker=dict(size=10, color="orange", symbol="diamond"),
                name="Seed point",
            )
        )

    # Grasp
    q = reconstruct_q(pred_pose, pred_dofs).to(device)

    # Grasp origin
    trans = q[0, :3].cpu()
    fig.add_trace(
        go.Scatter3d(
            x=[trans[0].item()],
            y=[trans[1].item()],
            z=[trans[2].item()],
            mode="markers",
            marker=dict(size=8, color="green", symbol="diamond"),
            name="Grasp origin",
        )
    )

    # Hand geometry
    if hand_model is not None:
        try:
            n_actual_dofs = hand_model.revolute_joints_q_lower.shape[1]
            q_trimmed = q[:, : 9 + n_actual_dofs]
            q_clean = torch.nan_to_num(q_trimmed, nan=0.0)

            if use_mesh:
                vis_data = hand_model.get_plotly_data(q=q_clean, opacity=0.6)
                for j, mesh_trace in enumerate(vis_data):
                    mesh_trace.color = "red"
                    mesh_trace.name = "Predicted grasp" if j == 0 else None
                    mesh_trace.showlegend = j == 0
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
                        marker=dict(size=2, color="red", opacity=0.85),
                        name="Predicted grasp",
                    )
                )
        except Exception as exc:
            print(f"  [WARN] Could not render robot geometry: {exc}")

    prompt_line = prompt_text if prompt_text else "(prompt unavailable in grasp file)"
    fig.update_layout(
        title=(
            f"[{item_idx}/{total_items}] {gripper_id} | {scene_id} | {object_id}"
            f"<br><sup>{prompt_source}:{prompt_key} | Prompt: {prompt_line}</sup>"
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
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize grasps one-by-one with intended prompt text."
    )
    parser.add_argument(
        "--grasps",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more *_predicted_grasps.json files.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.path.join(_REPO_ROOT, "configs"),
        help="Path to Hydra configs directory (default: <repo_root>/configs).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of grasp entries shown.",
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
        "--filter-prompt-substring",
        default=None,
        help="Case-insensitive substring filter on prompt text.",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Render grippers as meshes instead of point clouds.",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not pause for Enter between figures.",
    )
    args = parser.parse_args()

    cfg = load_cfg(os.path.abspath(args.config_dir))
    dataset_basedir = to_absolute_path(cfg.dataset.dataset_basedir)

    hand_models_path = os.path.join(dataset_basedir, "hand_models.pt")
    if os.path.exists(hand_models_path):
        print("Loading hand models...")
        hand_models = torch.load(hand_models_path, weights_only=False, map_location=device)
    else:
        print(
            f"[WARN] hand_models.pt not found at {hand_models_path}; grasp geometry will be skipped."
        )
        hand_models = {}

    scene_pc_path = os.path.join(dataset_basedir, "gnn_scene_adj_point_clouds_new.pt")
    print("Loading scene point clouds...")
    scene_pc_adj = torch.load(scene_pc_path, weights_only=False, map_location="cpu")

    robot_name_mapping = dict(cfg.dataset.robot_name_mapping)

    entries = []
    for grasp_file in args.grasps:
        print(f"Reading {grasp_file}")
        data = load_json(grasp_file)

        required = ["scene_id", "object_id", "gripper_id", "pred_pose", "pred_dofs"]
        missing = [k for k in required if k not in data]
        if missing:
            print(f"  [WARN] Skipping {grasp_file}: missing keys {missing}")
            continue

        n = len(data["pred_pose"])
        for i in range(n):
            scene_id = _as_scalar_id(data["scene_id"][i])
            object_id = _as_scalar_id(data["object_id"][i])
            gripper_id = _as_scalar_id(data["gripper_id"][i])
            pred_pose = data["pred_pose"][i]
            pred_dofs = data["pred_dofs"][i]
            seed_pt = _safe_get_list_value(data, "seed_pt", i, default=None)

            prompt = _safe_get_list_value(data, "prompt", i, default="")
            prompt_source = _safe_get_list_value(data, "prompt_source", i, default="unknown")
            prompt_key = _safe_get_list_value(data, "prompt_key", i, default="entry")

            if args.filter_gripper and gripper_id != args.filter_gripper:
                continue
            if args.filter_scene and scene_id != args.filter_scene:
                continue
            if args.filter_object and object_id != args.filter_object:
                continue
            if (
                args.filter_prompt_substring
                and args.filter_prompt_substring.lower() not in (prompt or "").lower()
            ):
                continue

            entries.append(
                {
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "gripper_id": gripper_id,
                    "pred_pose": pred_pose,
                    "pred_dofs": pred_dofs,
                    "seed_pt": seed_pt,
                    "prompt": prompt,
                    "prompt_source": prompt_source,
                    "prompt_key": prompt_key,
                }
            )

    if args.max_items is not None:
        entries = entries[: args.max_items]

    if not entries:
        print("No matching grasp entries to visualize.")
        return

    print(f"Visualizing {len(entries)} grasp entries one-by-one.")

    for idx, entry in enumerate(entries, start=1):
        scene_id = entry["scene_id"]
        if scene_id not in scene_pc_adj:
            print(
                f"  [WARN] Scene '{scene_id}' not found in scene point cloud file; skipping entry {idx}."
            )
            continue

        scene_features = scene_pc_adj[scene_id]["scene_features"]
        if scene_features.dim() == 3:
            scene_features = scene_features.squeeze(0)

        gripper_id = entry["gripper_id"]
        gripper_internal = robot_name_mapping.get(gripper_id, str(gripper_id).lower())
        hand_model = hand_models.get(gripper_internal)
        if hand_model is not None:
            hand_model = hand_model.to(device)

        print(
            f"[{idx}/{len(entries)}] {gripper_id} | {scene_id} | {entry['object_id']} | "
            f"{entry['prompt_source']}:{entry['prompt_key']}"
        )
        if entry["prompt"]:
            print(f"  Prompt: {entry['prompt']}")
        else:
            print("  Prompt: (unavailable)")

        fig = build_figure(
            gripper_id=gripper_id,
            scene_id=scene_id,
            object_id=entry["object_id"],
            prompt_text=entry["prompt"],
            prompt_source=entry["prompt_source"],
            prompt_key=entry["prompt_key"],
            seed_pt=entry["seed_pt"],
            pred_pose=entry["pred_pose"],
            pred_dofs=entry["pred_dofs"],
            scene_features=scene_features,
            hand_model=hand_model,
            use_mesh=args.mesh,
            item_idx=idx,
            total_items=len(entries),
        )
        fig.show()

        if not args.no_pause and idx < len(entries):
            try:
                input("Press Enter for next grasp (Ctrl+C to stop)... ")
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break

    print("Done.")


if __name__ == "__main__":
    main()
