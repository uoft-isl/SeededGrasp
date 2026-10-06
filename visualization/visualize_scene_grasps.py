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
Visualization script for scene grasp data.

Takes a scene grasp JSON file, samples a point cloud from the object .obj file,
and plots the grasps in the object frame.

Usage:
    python visualization/visualize_scene_grasps.py --grasp_file <path_to_grasp_json> [options]

Example:
    python visualization/visualize_scene_grasps.py \
        --grasp_file data/rgbd/scene_val/allegro/scene-multiple-0af9f30fcb159cdc-Allegro-2_of_Jenga_Classic_Game-feasible_grasps.json \
        --num_grasps 5 \
        --show_mesh
"""

import argparse
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
import trimesh as tm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scipy.spatial import cKDTree

from utils.general_utils import get_handmodel
from utils.math_utils import get_rot6d_from_quat

pio.renderers.default = "browser"


# Default paths - relative to the repo root; override with --object_dir/--data_dir as needed
DEFAULT_OBJECT_DIR = os.path.join(_REPO_ROOT, "data", "mgg", "objects")
DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "data", "rgbd")
DEFAULT_NUM_OBJ_POINTS = 2048

# DOF mapping from Isaac Sim ordering to SeededGrasp ordering
DOF_MAPPING = {
    "allegro": [0, 4, 8, 12, 2, 6, 10, 14, 3, 7, 11, 15, 1, 5, 9, 13],
    "franka_panda": [0, 1],
    "barrett": [0, 3, 6, 1, 4, 7, 2, 5],
    "robotiq_3finger": [2, 5, 8, 1, 4, 7, 10, 0, 3, 6, 9],
}

# Distinct colors for multiple grasps
GRASP_COLORS = [
    "red",
    "green",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "yellow",
    "pink",
    "brown",
    "lime",
]


def load_grasp_file(grasp_file: str) -> dict:
    """Load and parse grasp JSON file."""
    with open(grasp_file, "r") as f:
        grasp_data = json.load(f)
    return grasp_data


def load_object_mesh(object_id: str, object_dir: str) -> tm.Trimesh:
    """
    Load object mesh from GSO or YCB dataset.

    Args:
        object_id: Object identifier
        object_dir: Base directory containing GSO and YCB folders

    Returns:
        Trimesh object
    """
    # Check if object is GSO or YCB
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
        raise FileNotFoundError(f"Object mesh not found at: {obj_mesh_path}")

    print(f"Loading object mesh from: {obj_mesh_path}")
    return tm.load(obj_mesh_path)


def sample_object_point_cloud(mesh: tm.Trimesh, num_points: int) -> np.ndarray:
    """
    Sample point cloud from object mesh surface.

    Args:
        mesh: Trimesh object
        num_points: Number of points to sample

    Returns:
        Point cloud array of shape (num_points, 3)
    """
    return mesh.sample(num_points)


def build_grasp_q(pose: list, dofs: list, gripper_id: str) -> torch.Tensor:
    """
    Build the full q vector for a grasp (position + rot6d + dofs).

    Args:
        pose: Grasp pose [x, y, z, qx, qy, qz, qw] in object frame
        dofs: Joint DOF values
        gripper_id: Gripper identifier for DOF mapping

    Returns:
        q tensor of shape (1, 9 + num_dofs)
    """
    grasp_pose = torch.tensor(pose, dtype=torch.float32)
    grasp_dofs_raw = torch.tensor(dofs, dtype=torch.float32)

    # Apply DOF mapping to reorder from Isaac Sim ordering to SeededGrasp ordering
    dof_indices = DOF_MAPPING.get(gripper_id.lower(), list(range(len(grasp_dofs_raw))))
    grasp_dofs = grasp_dofs_raw[dof_indices]

    # Extract position and quaternion
    grasp_position = grasp_pose[:3]
    grasp_rotation_quat = grasp_pose[3:7]  # [w, x, y, z] format (scalar-first)

    # Note: The quaternion in the JSON is already in wxyz format (scalar-first),
    # which is what transforms3d expects, so no conversion is needed

    # Build full q vector: position (3) + rot6d (6) + dofs
    rot6d = torch.tensor(get_rot6d_from_quat(grasp_rotation_quat.numpy()), dtype=torch.float32)
    grasp_q = torch.cat((grasp_position, rot6d, grasp_dofs), dim=0).unsqueeze(0)

    return grasp_q


def find_closest_object_points(obj_pc: np.ndarray, gripper_pc: np.ndarray, k: int = 64) -> tuple:
    """
    Find the k closest object points to the gripper surface.

    Args:
        obj_pc: Object point cloud (N, 3)
        gripper_pc: Gripper surface point cloud (M, 3)
        k: Number of closest points to find

    Returns:
        Tuple of (closest_points, closest_indices)
    """
    # Build KD-tree for gripper points
    tree = cKDTree(gripper_pc)

    # For each object point, find distance to nearest gripper point
    distances, _ = tree.query(obj_pc, k=1)

    # Get indices of k smallest distances
    k_actual = min(k, len(obj_pc))
    closest_indices = np.argpartition(distances, k_actual - 1)[:k_actual]
    closest_indices = closest_indices[np.argsort(distances[closest_indices])]

    return obj_pc[closest_indices], closest_indices


def find_closest_points_to_center(
    obj_pc: np.ndarray,
    center: np.ndarray,
    k: int = 64,
    random_sample: bool = False,
    num_random: int = 32,
) -> tuple:
    """
    Find the k closest object points to a center point (gripper translation),
    then optionally randomly sample from them.

    Args:
        obj_pc: Object point cloud (N, 3)
        center: Center point (3,)
        k: Number of closest points to find
        random_sample: If True, randomly sample num_random points from the k closest
        num_random: Number of random points to sample if random_sample is True

    Returns:
        Tuple of (closest_points, closest_indices)
    """
    # Calculate distances from each object point to center
    distances = np.linalg.norm(obj_pc - center, axis=1)

    # Get indices of k smallest distances
    k_actual = min(k, len(obj_pc))
    closest_indices = np.argpartition(distances, k_actual - 1)[:k_actual]

    if random_sample:
        # Randomly sample num_random points from the k closest
        num_to_sample = min(num_random, len(closest_indices))
        sampled_idx = np.random.choice(len(closest_indices), num_to_sample, replace=False)
        closest_indices = closest_indices[sampled_idx]
    else:
        # Sort by distance
        closest_indices = closest_indices[np.argsort(distances[closest_indices])]

    return obj_pc[closest_indices], closest_indices


def visualize_grasps(
    grasp_data: dict,
    obj_pc: np.ndarray,
    hand_model,
    num_grasps: int = 5,
    show_mesh: bool = False,
    use_stable_poses: bool = True,
    grasp_indices: list = None,
    axis_range: float = 0.3,
    show_closest_points: bool = True,
    num_closest_points: int = 64,
    show_closest_to_center: bool = True,
    num_closest_to_center: int = 64,
) -> go.Figure:
    """
    Create Plotly visualization of object point cloud with grasp poses.

    Args:
        grasp_data: Parsed grasp JSON data
        obj_pc: Object point cloud (N, 3)
        hand_model: Hand model for visualization
        num_grasps: Number of grasps to visualize
        show_mesh: If True, show gripper mesh; if False, show point cloud
        use_stable_poses: If True, use stable_poses; if False, use pose
        show_closest_points: If True, highlight closest object points to gripper surface
        num_closest_points: Number of closest points to gripper surface to highlight
        show_closest_to_center: If True, highlight closest object points to gripper center
        num_closest_to_center: Number of closest points to gripper center to highlight
        grasp_indices: Specific grasp indices to visualize (overrides num_grasps)
        axis_range: Range for plot axes (symmetric around origin)

    Returns:
        Plotly Figure object
    """
    gripper_id = grasp_data["gripper_id"]
    object_id = grasp_data["object_id"]
    scene_id = grasp_data["scene_id"]

    # Select poses and dofs
    if use_stable_poses and "stable_poses" in grasp_data:
        poses = grasp_data["stable_poses"]
        dofs = grasp_data.get("stable_dofs", grasp_data.get("dofs", []))
    else:
        poses = grasp_data["pose"]
        dofs = grasp_data.get("dofs", [])

    total_grasps = min(len(poses), len(dofs))

    # Determine which grasps to visualize
    if grasp_indices is not None:
        indices = [i for i in grasp_indices if i < total_grasps]
    else:
        indices = list(range(min(num_grasps, total_grasps)))

    print(f"Visualizing {len(indices)} grasps out of {total_grasps} available")

    fig = go.Figure()

    # Plot object point cloud (in object frame, centered at origin)
    fig.add_trace(
        go.Scatter3d(
            x=obj_pc[:, 0],
            y=obj_pc[:, 1],
            z=obj_pc[:, 2],
            mode="markers",
            marker=dict(size=2, color="blue", opacity=0.7),
            name=f"Object: {object_id}",
        )
    )

    # Add origin marker
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=8, color="black", symbol="x"),
            name="Origin",
        )
    )

    # Plot each grasp
    for i, grasp_idx in enumerate(indices):
        color = GRASP_COLORS[i % len(GRASP_COLORS)]

        # Build q vector for this grasp
        grasp_q = build_grasp_q(poses[grasp_idx], dofs[grasp_idx], gripper_id)

        # Get gripper surface points for distance computation
        robot_features = hand_model.get_surface_points(
            q=grasp_q.to(torch.device("cuda")), downsample=True
        )
        gripper_pc = robot_features.squeeze().cpu().numpy()

        grasp_position = grasp_q[0, :3]

        # Find and visualize closest object points to gripper surface
        if show_closest_points:
            closest_pts, closest_idx = find_closest_object_points(
                obj_pc, gripper_pc, k=num_closest_points
            )
            fig.add_trace(
                go.Scatter3d(
                    x=closest_pts[:, 0],
                    y=closest_pts[:, 1],
                    z=closest_pts[:, 2],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=1.0, symbol="diamond"),
                    name=f"Closest to surface (Grasp {grasp_idx})",
                )
            )

        # Find and visualize closest object points to gripper center
        if show_closest_to_center:
            center_pts, center_idx = find_closest_points_to_center(
                obj_pc,
                grasp_position.cpu().numpy(),
                k=num_closest_to_center,
                random_sample=True,
                num_random=32,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=center_pts[:, 0],
                    y=center_pts[:, 1],
                    z=center_pts[:, 2],
                    mode="markers",
                    marker=dict(size=5, color=color, opacity=0.9, symbol="circle"),
                    name=f"Closest to center (Grasp {grasp_idx})",
                )
            )

        if show_mesh:
            # Show gripper mesh (slower but more detailed)
            vis_data = hand_model.get_plotly_data(q=grasp_q.to(torch.device("cuda")), opacity=0.6)
            for j, d in enumerate(vis_data):
                d.name = f"Grasp {grasp_idx}" if j == 0 else None
                d.showlegend = j == 0
                fig.add_trace(d)
        else:
            # Show gripper as point cloud (faster)
            fig.add_trace(
                go.Scatter3d(
                    x=gripper_pc[:, 0],
                    y=gripper_pc[:, 1],
                    z=gripper_pc[:, 2],
                    mode="markers",
                    marker=dict(size=2, color=color, opacity=0.8),
                    name=f"Grasp {grasp_idx}",
                )
            )

        # Plot grasp position marker
        fig.add_trace(
            go.Scatter3d(
                x=[grasp_position[0].item()],
                y=[grasp_position[1].item()],
                z=[grasp_position[2].item()],
                mode="markers",
                marker=dict(size=6, color=color, symbol="diamond"),
                name=f"Pos {grasp_idx}",
                showlegend=False,
            )
        )

    # Layout
    fig.update_layout(
        title=f"Scene: {scene_id}<br>Object: {object_id} | Gripper: {gripper_id}<br>Showing {len(indices)} grasps",
        scene=dict(
            xaxis=dict(range=[-axis_range, axis_range], title="X"),
            yaxis=dict(range=[-axis_range, axis_range], title="Y"),
            zaxis=dict(range=[-axis_range, axis_range], title="Z"),
            aspectmode="cube",
        ),
        width=1200,
        height=900,
        showlegend=True,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize scene grasps in object frame")
    parser.add_argument(
        "--grasp_file", "-g", type=str, required=True, help="Path to the grasp JSON file"
    )
    parser.add_argument(
        "--object_dir",
        "-o",
        type=str,
        default=DEFAULT_OBJECT_DIR,
        help="Directory containing GSO and YCB object meshes",
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Data directory containing URDF files",
    )
    parser.add_argument(
        "--num_points",
        "-p",
        type=int,
        default=DEFAULT_NUM_OBJ_POINTS,
        help="Number of points to sample from object mesh",
    )
    parser.add_argument(
        "--num_grasps", "-n", type=int, default=5, help="Number of grasps to visualize"
    )
    parser.add_argument(
        "--grasp_indices",
        "-i",
        type=int,
        nargs="+",
        default=None,
        help="Specific grasp indices to visualize (overrides --num_grasps)",
    )
    parser.add_argument(
        "--show_mesh",
        "-m",
        action="store_true",
        help="Show gripper mesh instead of point cloud (slower)",
    )
    parser.add_argument(
        "--show_closest_points",
        action="store_true",
        default=False,
        help="Highlight closest object points to gripper surface",
    )
    parser.add_argument(
        "--num_closest_points",
        "-k",
        type=int,
        default=64,
        help="Number of closest object points to gripper surface to highlight",
    )
    parser.add_argument(
        "--show_closest_to_center",
        action="store_true",
        default=True,
        help="Highlight closest object points to gripper center",
    )
    parser.add_argument(
        "--num_closest_to_center",
        "-c",
        type=int,
        default=64,
        help="Number of closest object points to gripper center to highlight",
    )
    parser.add_argument(
        "--use_raw_poses", action="store_true", help="Use raw poses instead of stable_poses"
    )
    parser.add_argument(
        "--axis_range",
        "-r",
        type=float,
        default=0.3,
        help="Axis range for visualization (symmetric around origin)",
    )
    parser.add_argument(
        "--save_html", type=str, default=None, help="Save visualization to HTML file"
    )

    args = parser.parse_args()

    # Load grasp data
    print(f"\n{'='*60}")
    print("Scene Grasp Visualization")
    print("=" * 60)
    print(f"Loading grasp file: {args.grasp_file}")

    grasp_data = load_grasp_file(args.grasp_file)

    gripper_id = grasp_data["gripper_id"]
    object_id = grasp_data["object_id"]
    scene_id = grasp_data["scene_id"]

    print(f"Gripper: {gripper_id}")
    print(f"Object: {object_id}")
    print(f"Scene: {scene_id}")
    print(f"Total grasps available: {len(grasp_data.get('pose', []))}")

    # Load object mesh and sample point cloud
    print(f"\n{'='*60}")
    print("Loading Object Mesh")
    print("=" * 60)

    obj_mesh = load_object_mesh(object_id, args.object_dir)
    obj_pc = sample_object_point_cloud(obj_mesh, args.num_points)
    print(f"Sampled {obj_pc.shape[0]} points from object mesh")

    # Load hand model
    print(f"\n{'='*60}")
    print("Loading Hand Model")
    print("=" * 60)

    hand_model = get_handmodel(
        gripper_id.lower(), 1, torch.device("cuda"), hand_scale=1.0, data_dir=args.data_dir
    )
    print(f"Loaded hand model: {gripper_id.lower()}")

    # Create visualization
    print(f"\n{'='*60}")
    print("Creating Visualization")
    print("=" * 60)

    fig = visualize_grasps(
        grasp_data=grasp_data,
        obj_pc=obj_pc,
        hand_model=hand_model,
        num_grasps=args.num_grasps,
        show_mesh=args.show_mesh,
        use_stable_poses=not args.use_raw_poses,
        grasp_indices=args.grasp_indices,
        axis_range=args.axis_range,
        show_closest_points=args.show_closest_points,
        num_closest_points=args.num_closest_points,
        show_closest_to_center=args.show_closest_to_center,
        num_closest_to_center=args.num_closest_to_center,
    )

    # Save or show
    if args.save_html:
        fig.write_html(args.save_html)
        print(f"Saved visualization to: {args.save_html}")
    else:
        fig.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
