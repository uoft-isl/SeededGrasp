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

import json
import os
import sys
import time

import hydra
import numpy as np
import plotly.graph_objects as go
import torch
from hydra.utils import instantiate, to_absolute_path
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import plotly.io as pio
import pypose as pp

import models.flow

pio.renderers.default = "browser"

from utils import math_utils
from utils.general_utils import get_handmodel
from utils.normalization import normalize_pc, normalize_q, unnormalize_q
from utils_data.augmentors import sample_pc_noise_random
from utils_data.rgbd_scene_dataset import (
    apply_gripper_rotation_alignment_with_transforms,
    build_gripper_alignment_transforms,
    collate_fn_rgbd,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mixed_eulers_method(x, time_steps, model, params, w=1.0):
    # Need to throw in one function to avoid multiple predictions
    with torch.no_grad():
        delta_t = time_steps[1] - time_steps[0]
        x_trans = x[:, :3]
        x_rot_9d = math_utils.robust_compute_rotation_matrix_from_ortho6d(x[:, 3:9])
        x_joints = x[:, 9:]

        drop_all = torch.ones(x.shape[0], dtype=torch.bool, device=device) if w != 1.0 else None

        all_x = [x]
        for t_i in range(len(time_steps)):
            t = time_steps[t_i]
            t_vec = torch.tensor([t], dtype=torch.float32).to(device).unsqueeze(0)
            u_theta = model(x, t_vec, *params)
            if w != 1.0:
                # Get unconditioned vector field by dropping all conditioning
                u_theta_uncond = model(x, t_vec, *params, drop_all)
                # Combine conditioned and unconditioned vector fields (CFG)
                u_theta = w * u_theta + (1.0 - w) * u_theta_uncond

            u_theta_trans = u_theta[:, :3]
            u_theta_rot = u_theta[:, 3:6]
            u_theta_so3 = pp.so3(u_theta_rot)
            u_theta_joints = u_theta[:, 6:]

            x_trans = x_trans + delta_t * u_theta_trans
            x_rot_9d = x_rot_9d @ math_utils.matrix_from_quat(
                pp.so3(delta_t * u_theta_so3).Exp()
            )  # TODO: definitely should optimize (keep as quat)
            x_joints = x_joints + delta_t * u_theta_joints

            x_rot_6d = x_rot_9d[:, :, :2].transpose(1, 2).reshape(x.shape[0], -1)
            x = torch.cat([x_trans, x_rot_6d, x_joints], dim=1)

            if t_i % 1 == 0:
                all_x.append(x)
        all_x.append(x)
        return all_x


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    # Load model
    robot_name_list = []
    for r in cfg.dataset.robot_name_list:
        robot_name_list.append(cfg.dataset.robot_name_mapping[r])
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
    model = models.flow.SeededGraspFlow(
        cfg=cfg, robot_models=robot_models, robot_models_path=robot_models_path
    ).to(device)
    model.eval()

    # Load weights
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
    print(latest_log_dir)
    log_dir = os.path.join(log_basedir, latest_log_dir)
    weight_dir = os.path.join(log_dir, cfg.train.weights_dir)
    model_path = os.path.join(weight_dir, cfg.train.weights_name)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Load datasets
    dataset_basedir = to_absolute_path(cfg.dataset.dataset_basedir)
    gripper_alignment_rot6d = getattr(cfg.dataset, "gripper_alignment_rot6d", None)
    canonical_rot6d_cfg = getattr(cfg.dataset, "canonical_rot6d", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    dataset = instantiate(
        cfg.dataset.dataloader,
        dataset_basedir=dataset_basedir,
        mode="validate",
        robot_name_list=robot_name_list,
        pad_q=cfg.dataset.pad_q,
        gripper_alignment_rot6d=gripper_alignment_rot6d,
        canonical_rot6d=canonical_rot6d_cfg,
    )

    dataloader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn_rgbd
    )

    alignment_transforms = build_gripper_alignment_transforms(
        gripper_alignment_rot6d=gripper_alignment_rot6d,
        canonical_rot6d=canonical_rot6d_cfg,
    )
    use_rotation_alignment = bool(gripper_alignment_rot6d)
    warned_unknown_grippers = set()
    canonical_rot6d = torch.tensor(
        canonical_rot6d_cfg, dtype=torch.float32, device=device
    ).unsqueeze(0)
    if canonical_rot6d.shape[1] != 6:
        raise ValueError("cfg.dataset.canonical_rot6d must contain 6 values")

    # Inference loop
    delta_t = 0.20  # Time step for simulation
    time_steps = torch.tensor(
        np.linspace(0, 1.0 - delta_t, int((1.0) / delta_t)), dtype=torch.float32
    ).to(device)
    counter = 0
    grasps_export = {
        "scene_id": [],
        "object_id": [],
        "gripper_id": [],
        "pred_pose": [],
        "pred_dofs": [],
    }
    for i, sample in enumerate(dataloader):
        start_time = time.time()
        # Prepare input data
        (
            _,  # field_names
            q,
            robot_adj,
            robot_features,
            rest_pose,
            obj_adj,  # scene_adj used as obj_adj
            obj_features,  # scene_features used as obj_features
            robot_name,
            object_name,
            scene_name,
            seed_pt,  # Pre-selected first closest point (B, 1, 3)
        ) = sample

        # Handle obj_adj - it may be a list of sparse tensors from collate_fn_rgbd
        if isinstance(obj_adj, list):
            # Transfer sparse adj matrices to GPU, then convert to batched dense
            obj_adj = torch.stack([adj.to(device).to_dense() for adj in obj_adj], dim=0)
        else:
            obj_adj = obj_adj.to(device)
        obj_features = obj_features.to(device)
        q = q.to(device)
        x = model.cond_prob_path.sample_x_init(q)
        seed_pt = seed_pt.to(device)
        mask = torch.isnan(q)
        if use_rotation_alignment:
            x = x.clone()
            x[:, :3] = seed_pt[:, 0, :]
            x[:, 3:9] = canonical_rot6d.expand(x.shape[0], -1)
            print(canonical_rot6d.expand(x.shape[0], -1))
            x[:, 9:] = 0.0
            if robot_name[0].lower() == "allegro":
                # Preserve the existing Allegro joint warm-start while keeping canonical rotation.
                allegro_init_joints = torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.75],
                    dtype=x.dtype,
                    device=device,
                ).unsqueeze(0)
                upper = min(9 + allegro_init_joints.shape[1], x.shape[1])
                x[:, 9:upper] = allegro_init_joints[:, : upper - 9]
        else:
            if robot_name[0].lower() == "franka_panda":
                x = torch.cat(
                    [
                        seed_pt[:, 0, :],
                        torch.tensor([-1, 0, 0, 0, 1, 0, 0.0, 0.0]).unsqueeze(0).to(device),
                        torch.zeros_like(x[:, 11:]).to(device),
                    ],
                    dim=1,
                )
            elif robot_name[0].lower() == "allegro":
                x = torch.cat(
                    [
                        seed_pt[:, 0, :],
                        torch.tensor(
                            [
                                0,
                                0,
                                -1,
                                -1,
                                0,
                                0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.5,
                                0.0,
                                0.75,
                            ]
                        )
                        .unsqueeze(0)
                        .to(device),
                        torch.zeros_like(x[:, 24:]).to(device),
                    ],
                    dim=1,
                )
            else:
                x = torch.cat(
                    [
                        seed_pt[:, 0, :],
                        torch.tensor([-1, 0, 0, 0, 0, -1, 0.0, 0.0]).unsqueeze(0).to(device),
                        torch.zeros_like(x[:, 11:]).to(device),
                    ],
                    dim=1,
                )
        x = normalize_q(cfg, x, robot_name)
        print(robot_name, object_name, scene_name)
        print(x)
        if cfg.dataset.augment_pc:
            random_pc_noise = sample_pc_noise_random(
                q.shape[0], obj_features.shape[1], device=device
            )
            obj_features = obj_features + random_pc_noise
        if cfg.dataset.normalize_q:
            q = normalize_q(cfg, q, robot_name)
            obj_features_norm = normalize_pc(cfg, obj_features)
            seed_pt = normalize_pc(cfg, seed_pt)
        else:
            obj_features_norm = obj_features

        # Convert robot names to indices for model compatibility
        robot_name_idx = torch.tensor(
            [model.robot_name_to_idx[rn] for rn in robot_name], device=device
        )
        params = (obj_features_norm, obj_adj, robot_name_idx, seed_pt)
        x[mask] = 0.0
        all_x = mixed_eulers_method(x, time_steps, model, params, w=cfg.model.w)
        print(f"Processed sample in {time.time() - start_time:.2f} seconds")

        if cfg.dataset.normalize_q:
            for i in range(len(all_x)):
                all_x[i] = unnormalize_q(cfg, all_x[i], robot_name)

        if use_rotation_alignment:
            for xi in range(len(all_x)):
                all_x[xi] = apply_gripper_rotation_alignment_with_transforms(
                    q=all_x[xi],
                    gripper_id=robot_name[0],
                    transforms=alignment_transforms,
                    inverse=True,
                    warned_unknown_grippers=warned_unknown_grippers,
                )

        if cfg.dataset.pad_q != 0:
            # Find mask
            mask = ~torch.isnan(q)
            for xi in range(len(all_x)):
                all_x[xi] = all_x[xi][mask].unsqueeze(0).cpu()

        obj_features = obj_features.cpu()

        # Unnormalize seed_pt for plotting (to match obj_features coordinate system)
        if cfg.dataset.normalize_q:
            # Reverse normalization: original = trans_limits * (normalized + 1) - trans_limits
            seed_pt_plot = seed_pt.clone().cpu()
            trans_limits = cfg.dataset.trans_limits
            for i in range(3):
                seed_pt_plot[:, :, i] = trans_limits * (seed_pt_plot[:, :, i] + 1) - trans_limits
        else:
            seed_pt_plot = seed_pt.cpu()

        # Plot the results
        if cfg.model.plot_grasps:
            fig = go.Figure()
            hand_model = get_handmodel(
                robot_name[0], 1, "cpu", hand_scale=1.0, data_dir=dataset_basedir
            )

            if not cfg.model.animate_path:
                vis_data = hand_model.get_plotly_data(q=all_x[-1])
                for d in vis_data:
                    fig.add_trace(d)
            else:
                vis_data = hand_model.get_plotly_data(q=all_x[0])
                for d in vis_data:
                    fig.add_trace(d)

            fig.add_trace(
                go.Scatter3d(
                    x=obj_features.squeeze()[:, 0],
                    y=obj_features.squeeze()[:, 1],
                    z=obj_features.squeeze()[:, 2],
                    mode="markers",
                    marker=dict(size=1, color="red"),
                    name="Object Contacts",
                )
            )

            seed_pt_np = seed_pt_plot.numpy()
            fig.add_trace(
                go.Scatter3d(
                    x=[seed_pt_np.squeeze()[0]],
                    y=[seed_pt_np.squeeze()[1]],
                    z=[seed_pt_np.squeeze()[2]],
                    mode="markers",
                    marker=dict(size=4, color="blue"),
                    name="Seed Point",
                )
            )

            # Find 16 nearest points to seed point (same method as in flow.py)
            # Use unnormalized seed_pt for distance calculation in original coordinate space
            dists = torch.cdist(seed_pt_plot, obj_features)
            knn_idx = dists.topk(k=16, largest=False).indices.squeeze(1)
            neighbour_pts = torch.gather(obj_features, 1, knn_idx.unsqueeze(-1).expand(-1, -1, 3))

            # Calculate centroid of nearest neighbors
            centroid = neighbour_pts.mean(dim=1).squeeze().cpu().numpy()

            # Find and plot 16 nearest points to the centroid
            centroid_tensor = torch.tensor(centroid).unsqueeze(0).unsqueeze(0)
            dists_centroid = torch.cdist(centroid_tensor, obj_features)
            knn_idx_centroid = dists_centroid.topk(k=16, largest=False).indices.squeeze(1)
            neighbour_pts_centroid = torch.gather(
                obj_features, 1, knn_idx_centroid.unsqueeze(-1).expand(-1, -1, 3)
            )
            neighbour_pts_centroid_np = neighbour_pts_centroid.squeeze().cpu().numpy()

            fig.add_trace(
                go.Scatter3d(
                    x=neighbour_pts_centroid_np[:, 0],
                    y=neighbour_pts_centroid_np[:, 1],
                    z=neighbour_pts_centroid_np[:, 2],
                    mode="markers",
                    marker=dict(size=3, color="purple"),
                    name="16 Nearest to Centroid",
                )
            )

            if not cfg.model.animate_path:
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(range=[-1, 1]),
                        yaxis=dict(range=[-1, 1]),
                        zaxis=dict(range=[-1, 1]),
                        aspectmode="cube",  # keep x=y=z aspect ratio
                    ),
                    width=800,
                    height=800,
                )
            else:
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(range=[-1, 1]),
                        yaxis=dict(range=[-1, 1]),
                        zaxis=dict(range=[-1, 1]),
                        aspectmode="cube",  # keep x=y=z aspect ratio
                    ),
                    width=800,
                    height=800,
                    updatemenus=[
                        dict(
                            type="buttons",
                            buttons=[
                                dict(
                                    label="Play",
                                    method="animate",
                                    args=[
                                        None,
                                        {
                                            "frame": {"duration": 250, "redraw": True},
                                            "fromcurrent": True,
                                        },
                                    ],
                                ),
                                dict(
                                    label="Pause",
                                    method="animate",
                                    args=[
                                        [None],
                                        {
                                            "frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate",
                                        },
                                    ],
                                ),
                            ],
                        )
                    ],
                )

                frames = []
                for xi in range(len(all_x)):
                    vis_data = hand_model.get_plotly_data(q=all_x[xi])
                    frame = go.Frame(data=[*vis_data])
                    frames.append(frame)
                fig.frames = frames

            fig.show()

        if cfg.model.export_grasps:
            final_q = all_x[-1].squeeze(0).cpu().numpy()
            pred_pose = final_q[:9]
            pred_dofs = final_q[9:]
            grasps_export["scene_id"].append(scene_name[0])
            grasps_export["object_id"].append(object_name)
            grasps_export["gripper_id"].append(robot_name)
            grasps_export["pred_pose"].append(pred_pose.tolist())
            grasps_export["pred_dofs"].append(pred_dofs.tolist())

        counter += 1
        if counter >= cfg.model.gen_samples:  # Limit to 10 samples for demonstration
            break

    if cfg.model.export_grasps:
        export_path = os.path.join(log_dir, "predicted_grasps.json")
        with open(export_path, "w") as f:
            json.dump(grasps_export, f, indent=4)
        print(f"Exported predicted grasps to {export_path}")


if __name__ == "__main__":
    main()
