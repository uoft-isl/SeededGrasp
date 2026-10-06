"""Sample surface point clouds of grippers and objects.

Grippers are posed with every joint at the middle of its limits. For each asset three PLY files are
written: the full sample (--num_total_samples), a farthest-point downsample (--num_final_samples) and
the downsample with Gaussian noise (--noise).
    <output_dir>/grippers/<gripper>/<gripper>_{full,downsampled,<noise>_noised_downsampled}_point_cloud.ply
    <output_dir>/objects/<object>/<object>_{...}_point_cloud.ply
"""
import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scene_utils import (create_world, import_gripper_with_physics, get_matching_prim_paths, get_mesh_vertices,
                         face_sampling_from_mesh, farthest_point_sampling, add_noise_to_points, save_point_cloud_to_ply)

def parse_args():
    parser = argparse.ArgumentParser(description="Sample point cloud from prims in scene")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the point cloud data")
    parser.add_argument("--gripper_dir", type=str, default=None, help="Directory containing gripper USD files")
    parser.add_argument("--object_dir", type=str, default=None, help="Directory containing object USD files")
    parser.add_argument("--noise", type=float, default=0.005, help="Standard deviation of Gaussian noise to add to point cloud") # Empirically determined 0.005 to be reasonable
    parser.add_argument("--num_total_samples", type=int, default=60000, help="Total number of samples to take from the mesh")
    parser.add_argument("--num_final_samples", type=int, default=2000, help="Number of samples to keep after downsampling")
    parser.add_argument("--device", type=int, default=0, help="GPU to run the simulation on")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    return parser.parse_args()

def set_joints_to_mid_limits(articulation_view):
    dof_limits = articulation_view.get_dof_limits()
    #dof_limits = dof_limits.squeeze()
    mid_limits = (dof_limits[:, :, 0] + dof_limits[:, :, 1]) / 2.0
    articulation_view.set_joint_positions(mid_limits)

def get_articulation_mesh_values(world, usd_path, gripper_info):
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.prims.rigid_prim import RigidPrimView
    from omni.isaac.core.utils.transformations import tf_matrix_from_pose
    usd_name = usd_path.split("/")[-1].split(".")[0]

    # Spawn gripper w/ joints at mid limits
    prim_utils.define_prim(f"/World/{usd_name}")
    gripper_art, gripper_view = import_gripper_with_physics(world=world, 
                                                            work_path=f"/World/{usd_name}", 
                                                            usd_path=usd_path,
                                                            EF_axis=gripper_info["EF_axis"])
    
    art_link_paths = get_matching_prim_paths(world, rf"^\/World\/{usd_name}\/gripper\/(?!.*root_joint)[^\/]+$")
    art_rigid_view = world.scene.add(
        RigidPrimView(
            prim_paths_expr=art_link_paths,
            name=f"{usd_name}_gripper_view",
        )
    )
    world.reset()
    gripper_view.initialize(world.physics_sim_view)
    art_rigid_view.initialize(world.physics_sim_view)
    set_joints_to_mid_limits(gripper_view)

    # Get world poses of the articulation links
    link_world_poses = {}
    world_poses = art_rigid_view.get_world_poses()
    art_link_paths = art_rigid_view.prim_paths
    for path, idx in zip(art_link_paths, range(len(world_poses[0]))):
        link_world_poses[path] = tf_matrix_from_pose(world_poses[0][idx], world_poses[1][idx])

    # Get mesh vertices and faces for entire gripper
    # Note: visual meshes are used since they capture the full gripper geometry.
    # Note: meshes are looked up by path since stage.Traverse() does not reach them (instanced).
    vertices, face_vertex_counts, face_vertex_indices = np.array([]), np.array([]), np.array([])
    for art_link in art_link_paths:
        mesh_idx = 0
        while True:
            mesh_path = art_link + f"/visuals/mesh_{mesh_idx}"
            temp_vert, temp_face_count, temp_face_index = get_mesh_vertices(world, mesh_path, link_world_poses[art_link])
            if temp_vert is None:
                break

            temp_face_index += len(vertices) # Need to re-index to be global across all meshes in articulation
            vertices = np.concatenate((vertices, temp_vert), axis=0) if vertices.size > 0 else temp_vert
            face_vertex_counts = np.concatenate((face_vertex_counts, temp_face_count), axis=0) if face_vertex_counts.size > 0 else temp_face_count
            face_vertex_indices = np.concatenate((face_vertex_indices, temp_face_index), axis=0) if face_vertex_indices.size > 0 else temp_face_index
            mesh_idx += 1
    if vertices.size == 0:
        raise ValueError(f"No valid meshes found for articulation at {usd_path}")
    return vertices, face_vertex_counts, face_vertex_indices, link_world_poses

def get_obj_mesh_values(world, usd_path):
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.prims.geometry_prim import GeometryPrim
    usd_name = usd_path.split("/")[-1].split(".")[0]

    # Spawn object in default pose
    add_reference_to_stage(usd_path=usd_path, prim_path=f"/World/objects/a_{usd_name}")
    world.scene.add(
        GeometryPrim(
            prim_path=f"/World/objects/a_{usd_name}",
            name=usd_name
        )
    )

    # Get object mesh vertices and faces (assuming objs don't have more than 1 mesh)
    # Note: visual meshes are used since they capture the full object geometry.
    vertices, face_vertex_counts, face_vertex_indices = get_mesh_vertices(world, f"/World/objects/a_{usd_name}/base_link/visuals/mesh_0", np.eye(4))
    if vertices is None or face_vertex_counts is None or face_vertex_indices is None or vertices.size == 0:
        raise ValueError(f"No valid mesh found for object at {usd_path}")
    return vertices, face_vertex_counts, face_vertex_indices


if __name__=="__main__":
    args = parse_args()
    output_dir = args.output_dir
    gripper_dir = args.gripper_dir
    object_dir = args.object_dir
    noise_std = args.noise
    num_total_samples = args.num_total_samples
    num_final_samples = args.num_final_samples
    os.makedirs(output_dir, exist_ok=True)
    
    ######################### Launch Isaac Sim #########################
    from omni.isaac.kit import SimulationApp
    config = {
        "headless": args.headless,
        "max_bounces": 0,
        "fast_shutdown": True,
        "max_specular_transmission_bounces": 0,
        "physics_gpu": args.device,
        "active_gpu": args.device,
        "multi_gpu": True
    }
    simulation_app = SimulationApp(config)
    import omni.isaac.core.utils.stage as stage_utils

    ######################### Get Point Clouds #########################
    # Fetch relevant usd files
    gripper_usd_files, object_usd_files = [], []
    if gripper_dir is not None:
        # path follows format args.gripper_dir/gripper_name/gripper_name/gripper_name.usd
        for gripper in os.listdir(gripper_dir):
            gripper_path = Path(f"{gripper_dir}/{gripper}/{gripper}/{gripper}.usd")
            if not gripper_path.exists():
                print(f"Gripper USD file {gripper_path} does not exist, skipping.")
                continue
            gripper_usd_files.append(gripper_path)
    if object_dir is not None:
        # path follows format args.object_dir/object_name/object_name.usd
        for obj in os.listdir(object_dir):
            obj_path = Path(f"{object_dir}/{obj}/{obj}.usd")
            if not obj_path.exists():
                print(f"Object USD file {obj_path} does not exist, skipping.")
                continue
            object_usd_files.append(obj_path)

    def save_point_clouds(points_dir, name, sampled_points, downsampled_points, noised_downsampled_points):
        os.makedirs(points_dir, exist_ok=True)
        save_point_cloud_to_ply(sampled_points, f"{points_dir}/{name}_full_point_cloud.ply")
        save_point_cloud_to_ply(downsampled_points, f"{points_dir}/{name}_downsampled_point_cloud.ply")
        save_point_cloud_to_ply(noised_downsampled_points, f"{points_dir}/{name}_{noise_std}_noised_downsampled_point_cloud.ply")

    def sample(vertices, face_vertex_counts, face_vertex_indices):
        sampled_points = face_sampling_from_mesh(vertices, face_vertex_counts, face_vertex_indices, num_samples=num_total_samples)
        downsampled_points = farthest_point_sampling(sampled_points, num_final_samples)
        return sampled_points, downsampled_points, add_noise_to_points(downsampled_points, noise_std)

    # For each usd file, load mesh, sample points, and save to PLY
    for gripper_usd in tqdm(gripper_usd_files, desc="Processing Grippers"):
        gripper_name = gripper_usd.stem
        world = create_world()
        try:
            gripper_info = json.load(open(os.path.join(gripper_dir, "gripper_isaac_info.json")))[gripper_name]
            vertices, face_vertex_counts, face_vertex_indices, _ = get_articulation_mesh_values(world, str(gripper_usd), gripper_info)
            save_point_clouds(f"{output_dir}/grippers/{gripper_name}", gripper_name, *sample(vertices, face_vertex_counts, face_vertex_indices))
        except Exception as e:
            print(f"Error processing gripper {gripper_name}: {e}")
        stage_utils.create_new_stage()
    
    for object_usd in tqdm(object_usd_files, desc="Processing Objects"):
        object_name = object_usd.stem
        world = create_world()
        try:
            vertices, face_vertex_counts, face_vertex_indices = get_obj_mesh_values(world, str(object_usd))
            save_point_clouds(f"{output_dir}/objects/{object_name}", object_name, *sample(vertices, face_vertex_counts, face_vertex_indices))
        except Exception as e:
            print(f"Error processing object {object_name}: {e}")
        stage_utils.create_new_stage()

    ######################### Clean Up #########################
    simulation_app.close()
