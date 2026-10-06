"""Save viewport screenshots of feasible / infeasible grasps produced by get_feasible_grasps.py.

For every scene-object pair, up to --num_screenshots grasps with result 1.0 and with result 0.0 are
rendered in the scene and saved as <output_dir>/<scene_id>-<gripper>-<object>_<idx>_<result>.png.
"""
import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scene_utils import load_grasps, debug_pause, import_gripper_with_physics, get_matching_prim_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize feasible grasps for objects in a given scene")
    parser.add_argument("--scene_dir", type=str, default="./scenes", help="Directory containing scene folders (from gen_clutter_scene.py)")
    parser.add_argument("--gripper_dir", type=str, default="./grippers", help="Directory containing gripper files (USD)")
    parser.add_argument("--gripper", type=str, required=True, help="Name of the gripper to use")
    parser.add_argument("--grasp_dir", type=str, default="./grasps", help="Directory containing the output of get_feasible_grasps.py")
    parser.add_argument("--object_dir", type=str, default="./objects", help="Directory containing object files (USD)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save screenshots")
    parser.add_argument("--device", type=str, default="0", help="GPU to run the simulation on")
    parser.add_argument("--num_screenshots", type=int, default=10, help="Number of screenshots for each scene-object (good and bad each).")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    scene_dir = args.scene_dir
    gripper_dir = args.gripper_dir
    grasp_dir = args.grasp_dir
    object_dir = args.object_dir
    output_dir = args.output_dir
    gripper_name = args.gripper
    device = args.device
    num_screenshots = args.num_screenshots
    num_workers = 1
    os.makedirs(output_dir, exist_ok=True)

    # Check for at least one scene in the scene dir
    scene_files = list(map(str, Path(scene_dir).rglob("*.json")))
    if len(scene_files) == 0:
        raise ValueError(f"No scene files found in {scene_dir}.")
    # Check if specified gripper usd file exists
    gripper_file = os.path.join(gripper_dir, f"{gripper_name}/{gripper_name}/{gripper_name}.usd")
    if not os.path.exists(gripper_file):
        raise ValueError(f"Gripper file {gripper_file} does not exist. Please check the gripper name.")
    
    ######################### Launch Isaac Sim #########################
    from omni.isaac.kit import SimulationApp
    config = {
        "headless": True,
        "max_bounces": 0,
        "fast_shutdown": True,
        "max_specular_transmission_bounces": 0,
        "physics_gpu": device,
        "active_gpu": device,
        "multi_gpu": True
    }
    simulation_app = SimulationApp(config)

    # Omniverse imports
    from omni.isaac.core import World
    from omni.isaac.cloner import GridCloner
    import omni.isaac.core.utils.stage as stage_utils
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.prims.geometry_prim import GeometryPrim, GeometryPrimView
    from omni.isaac.core.prims.rigid_prim import RigidPrimView
    from omni.isaac.core.articulations import ArticulationView
    from omni.isaac.core.utils.transformations import pose_from_tf_matrix, tf_matrices_from_poses
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file, frame_viewport_prims

    ######################### World Setup #########################
    with open(os.path.join(gripper_dir, "gripper_isaac_info.json")) as f:
        gripper_info = json.load(f)[gripper_name]

    for scene_file in scene_files:
        print(f"\nLoading scene: {scene_file}")
        with open(scene_file, 'r') as f:
            scene_info = json.load(f)
        scene_id = Path(scene_file).stem

        # Create world
        stage_utils.create_new_stage()
        world = World(physics_dt=1/240.0, set_defaults=True)
        world.scene.add_default_ground_plane()
        world.reset()

        physicsContext = world.get_physics_context()
        physicsContext.set_solver_type("TGS")
        physicsContext.enable_gpu_dynamics(True)
        physicsContext.enable_stablization(True)
        world.reset()
        
        ######################### Setup workstations #########################
        # Add objects
        for i in scene_info.keys():
            if "camera" in i.lower():
                continue

            object_path = os.path.join(object_dir, i, f"{i}.usd")
            if not os.path.exists(object_path):
                raise ValueError(f"Object file {object_path} does not exist. Please check the object directory.")
            
            add_reference_to_stage(
                usd_path=os.path.abspath(object_path), 
                prim_path=f"/World/workstation_0/objects/a_{i}"
            )
            world.scene.add(
                GeometryPrim(
                    prim_path=f"/World/workstation_0/objects/a_{i}",
                    name=i,
                    position=scene_info[i]["position"],
                    orientation=scene_info[i]["orientation"],
                )
            )

        # Add gripper
        prim_utils.define_prim("/World/workstation_0/Robot")
        import_gripper_with_physics(world=world,
                                    work_path="/World/workstation_0/Robot",
                                    usd_path=gripper_file,
                                    EF_axis=gripper_info["EF_axis"])
        
        # Clone workstation
        cloner = GridCloner(spacing=2)
        target_paths = [f"/World/workstation_{i}" for i in range(num_workers)]
        cloner.clone(source_prim_path="/World/workstation_0", prim_paths=target_paths,
                     copy_from_source=True, replicate_physics=True, base_env_path="/World",
                     root_path="/World/workstation_")

        # Get views
        gripper_arts = world.scene.add(
            ArticulationView(
                prim_paths_expr="/World/workstation_*/Robot/gripper",
                reset_xform_properties=False,
                name="all_gripper_art_view"
            )
        )
        art_link_paths = get_matching_prim_paths(world, rf"^\/World\/workstation_.*\/Robot\/gripper\/(?!.*root_joint)[^\/]+$")
        art_geo_view = world.scene.add(
            GeometryPrimView(
                prim_paths_expr=art_link_paths,
                name=f"all_links_geo_view",
            )
        )

        ######################### Load Grasp Poses #########################
        # Get all objects in the scene
        all_object_prim_paths = prim_utils.find_matching_prim_paths("/World/workstation_0/objects/*")
        all_object_names = [path.split("/")[-1][2:] for path in all_object_prim_paths]

        # Iterate through objects
        for i, object_name in enumerate(all_object_names):
            print(f"\nProcessing object: {object_name} ({i+1}/{len(all_object_names)})")

            # Setup target object view
            target_object_view = world.scene.add(
                RigidPrimView(
                    prim_paths_expr=f"/World/workstation_*/objects/a_{object_name}/base_link",
                    name=f"target_{object_name}_view",
                )
            )

            # Load grasps
            grasps = load_grasps(grasp_dir, gripper_name, object_name, grasp_file=f"{scene_id}-{gripper_name}-{object_name}-feasible_grasps.json")
            if len(grasps["stable_poses"]) == 0:
                continue
            print(f"Loaded {len(grasps['stable_poses'])} grasps for {object_name}")

            # Divide grasps among workers
            stable_poses = np.array(grasps["stable_poses"])
            stable_dofs = np.array(grasps["stable_dofs"])
            results = np.array(grasps["result"])
            grasps_per_worker = int(np.ceil(len(stable_poses) / num_workers))
            assignment_idxs = np.arange(grasps_per_worker * num_workers)
            assignment_idxs[assignment_idxs >= len(stable_poses)] = -1
            assignment_idxs = assignment_idxs.reshape((grasps_per_worker, num_workers))

            # Iterate through assigned grasps
            num_good_shots = 0
            num_bad_shots = 0
            for j in tqdm(range(assignment_idxs.shape[0])):
                poses = stable_poses[assignment_idxs[j]]
                dofs = stable_dofs[assignment_idxs[j]]
                res = results[assignment_idxs[j]]

                if res[0] < 0.5: # Bad
                    if num_bad_shots >= num_screenshots:
                        continue
                    num_bad_shots += 1
                if res[0] > 0.5: # Good
                    if num_good_shots >= num_screenshots:
                        continue
                    num_good_shots += 1

                # Reset world and initialize new prims
                world.reset()
                target_object_view.initialize(world.physics_sim_view)
                gripper_arts.initialize(world.physics_sim_view)
                art_geo_view.initialize(world.physics_sim_view)

                # Load gripper positions
                gripper_arts.set_joint_positions(dofs)

                # Orient grippers to grasp poses
                target_object_trans, target_object_rot = target_object_view.get_world_poses()
                target_object_T = tf_matrices_from_poses(target_object_trans, target_object_rot)
                grasp_T = tf_matrices_from_poses(poses[:, :3], poses[:, 3:])
                desired_gripper_poses = [pose_from_tf_matrix(tf) for tf in np.matmul(target_object_T, grasp_T).astype(float)]
                gripper_arts.set_world_poses(np.array([p[0] for p in desired_gripper_poses]), np.array([p[1] for p in desired_gripper_poses]))

                vp_api = get_active_viewport()
                frame_viewport_prims(vp_api, [f"/World/workstation_0/objects", f"/World/workstation_0/Robot/gripper"])
                debug_pause(world=world, step=True, num_steps=200) # delay needed to take screenshot properly
                capture_viewport_to_file(
                    vp_api,
                    os.path.join(output_dir, f"{scene_id}-{gripper_name}-{object_name}_{j}_{res[0]}.png"),
                )
                print(f"Screenshot saved to {output_dir}")
                debug_pause(world=world, step=True, num_steps=200)

    ######################### Clean Up #########################
    simulation_app.close()
