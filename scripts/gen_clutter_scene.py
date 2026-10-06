"""Generate table-top scenes by dropping objects onto a ground plane and letting them settle.

Two modes:
    - default: every scene contains --num_objects distinct random objects (optionally a random number with --rand_obj_num).
    - --per_object: for every object, --num_scenes single-object scenes with random initial orientations are
      simulated, and near-duplicate resting poses are discarded (at most MAX_POSES_PER_OBJECT kept per object).

Each scene is saved to <output_dir>/scene-<object or "multiple">-<hash>/ as
    <scene>.usd  (full stage), <scene>.json (object name -> settled world pose), <scene>.png (viewport capture).
"""
import os
import sys
import time
import json
import hashlib
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scene_utils import debug_pause

MAX_POSES_PER_OBJECT = 6        # --per_object: unique resting poses kept per object
SIMILAR_POSE_DIST = 0.6         # m, poses closer than this are compared by rotation
SIMILAR_POSE_ANGLE = 30.0       # deg, ... and considered duplicates if within this angle
SIMILAR_POSE_Z_AXIS = 0.9       # ... or if they differ only by a rotation about (close to) the z axis

def parse_args():
    parser = argparse.ArgumentParser(description="Generate cluttered scene. Table top with x objects")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    parser.add_argument("--device", type=str, default="0", help="GPU to run the simulation on")
    parser.add_argument("--num_objects", type=int, default=5, help="Number of objects to spawn in the scene. Disregarded if --per_object is set")
    parser.add_argument("--rand_obj_num", action="store_true", help="Randomize number of objects per scene with num objects as upper bound")
    parser.add_argument("--objects_dir", type=str, default="./objects", help="Directory containing object files (USD)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the scene")
    parser.add_argument("--num_scenes", type=int, default=1, help="Number of scenes to generate (per object if --per_object is set)")
    parser.add_argument("--per_object", action="store_true", help="Generate single-object scenes for every object")
    parser.add_argument("--obj_list_path", type=str, default="", help="Path to txt file listing objs to be used. If empty, use all objs in objects_dir")
    return parser.parse_args()

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def gen_unique_hash():
    return hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]

if __name__=="__main__":
    args = parse_args()
    headless = args.headless
    num_objects = args.num_objects
    object_dir = args.objects_dir
    output_dir = args.output_dir
    num_scenes = args.num_scenes
    per_object = args.per_object

    # Check for at least one object in the object dir
    object_files = [os.path.abspath(f) for f in Path(object_dir).rglob("*.usd") if "instanceable_mesh" not in str(f)]
    if len(object_files) == 0:
        raise ValueError(f"No object files found in {object_dir}. Please add some USD files.")

    if args.obj_list_path != "":
        with open(args.obj_list_path, 'r') as f:
            obj_list = f.read().splitlines()
        filtered_object_files = []
        for obj_name in obj_list:
            matched_files = [f for f in object_files if os.path.basename(f).startswith(obj_name)]
            if len(matched_files) == 0:
                print(f"Warning: No matching file found for {obj_name} in {object_dir}")
            else:
                filtered_object_files.extend(matched_files)
        object_files = filtered_object_files
        if len(object_files) == 0:
            raise ValueError(f"No matching object files found from {args.obj_list_path} in {object_dir}. Please check the file names.")

    ######################### Launch Isaac Sim #########################
    from omni.isaac.kit import SimulationApp
    config = {
        "headless": headless,
        "max_bounces": 0,
        "fast_shutdown": True,
        "max_specular_transmission_bounces": 0,
        "physics_gpu": args.device,
        "active_gpu": args.device,
        "multi_gpu": True
    }
    simulation_app = SimulationApp(config)

    # Omniverse imports
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.prims import RigidPrimView
    from omni.isaac.core.prims.geometry_prim import GeometryPrim
    import omni.isaac.core.utils.stage as stage_utils
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file, frame_viewport_prims

    # Custom imports
    from sim_utils import add_light

    rng = np.random.default_rng()

    if per_object:
        existing_final_poses = [] # Final poses of the current object, used to filter too similar poses
        random_poses = R.random(num_scenes).as_quat() # Random orientations for the object
        num_scenes = num_scenes * len(object_files)
        num_objects = 1 # Only one object per scene in this case

    ######################### World Setup #########################
    for s in tqdm(range(num_scenes)):
        if per_object:
            if s % args.num_scenes == 0:
                existing_final_poses = [] # Reset for new object
            if len(existing_final_poses) >= MAX_POSES_PER_OBJECT:
                print(f"Skipping scene {s+1}: already have {MAX_POSES_PER_OBJECT} unique poses for {os.path.basename(object_files[s // args.num_scenes])}.")
                continue

        world = World(physics_dt=1/240.0, set_defaults=True)
        render = not headless

        # Initialize flat surface
        world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        world.reset()

        # Set physics options
        physicsContext = world.get_physics_context()
        physicsContext.set_solver_type("TGS")
        physicsContext.enable_gpu_dynamics(True)
        physicsContext.enable_stablization(True)
        physicsContext.set_gravity(-9.81)
        world.reset()

        if args.rand_obj_num:
            num_objects_scene = rng.integers(1, num_objects, endpoint=True)
        else:
            num_objects_scene = num_objects

        ######################### Spawn Objects / Generate Scene #########################
        # All object prims get an 'a_' prefix since Isaac Sim prim names cannot start with a digit
        if not per_object:
            if len(object_files) < num_objects_scene:
                raise ValueError(f"Not enough objects in {object_dir} to spawn {num_objects_scene} objects. Found {len(object_files)}.")

            # Objects are unique within a scene but can be reused across scenes
            scene_object_files = rng.choice(object_files, size=num_objects_scene, replace=False)
            for i, object_path in enumerate(scene_object_files):
                object_name = os.path.basename(object_path).split(".")[0]
                print(f"Adding object {i+1}/{num_objects_scene}: {object_name}")

                # Random orientation, objects stacked in a column above the ground plane
                orientation = R.random().as_quat()
                orientation = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
                obj_grid_size = 1 # Must be int
                xy_grid_interval = 0.30
                position = np.array([xy_grid_interval * ( i % obj_grid_size ) - xy_grid_interval * (obj_grid_size // 2),
                                    xy_grid_interval * (( i // obj_grid_size ) % obj_grid_size) - xy_grid_interval * (obj_grid_size // 2),
                                    0.3 + (xy_grid_interval * (i // obj_grid_size ** 2))])

                add_reference_to_stage(usd_path=object_path, prim_path=f"/World/objects/a_{object_name}")
                world.scene.add(
                    GeometryPrim(
                        prim_path=f"/World/objects/a_{object_name}",
                        name=object_name,
                        position=position,
                        orientation=orientation,
                    )
                )
        else:
            object_path = object_files[s // args.num_scenes]
            object_name = os.path.basename(object_path).split(".")[0]
            print(f"Adding object {s+1}/{num_scenes}: {object_name}")

            orientation = random_poses[s % args.num_scenes]
            orientation = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
            position = np.array([np.random.uniform(-0.2, 0.2),
                                 np.random.uniform(-0.2, 0.2),
                                 0.5])

            add_reference_to_stage(usd_path=object_path, prim_path=f"/World/objects/a_{object_name}")
            world.scene.add(
                GeometryPrim(
                    prim_path=f"/World/objects/a_{object_name}",
                    name=object_name,
                    position=position,
                    orientation=orientation,
                )
            )

        rigid_obj = world.scene.add(
            RigidPrimView(
                prim_paths_expr=f"/World/objects/*/base_link",
                track_contact_forces=True,
                prepare_contact_sensors=True,
            )
        )
        world.reset()
        rigid_obj.initialize(world.physics_sim_view)
        rigid_obj.set_velocities(np.zeros((num_objects_scene, 6)))

        if render:
            add_light()

        # Simulate long enough for the objects to fall and settle
        num_steps = 5.0/world.get_physics_dt()
        for _ in range(int(num_steps)):
            world.step(render=render)
        world.pause()

        ######################### Save Scene #########################
        all_object_names = [prim_p.split("/a_")[-1].split("/")[0] for prim_p in rigid_obj.prim_paths]

        # Check if final scene too similar to existing scenes for 1 object case
        too_similar = False
        if per_object:
            curr_obj_trans, curr_obj_rot = rigid_obj.get_world_poses()
            curr_obj_trans, curr_obj_rot = curr_obj_trans[0], curr_obj_rot[0]

            for pose in existing_final_poses:
                existing_trans = pose[:3]
                existing_rot = pose[3:]

                deg_diff = 2 * np.arccos(abs(np.clip(np.dot(existing_rot, curr_obj_rot), -1.0, 1.0))) * 180.0 / np.pi
                quat_diff = quaternion_multiply(curr_obj_rot, np.array([existing_rot[0], -existing_rot[1], -existing_rot[2], -existing_rot[3]]))
                diff_rot_axis = np.abs((quat_diff/np.linalg.norm(quat_diff[1:]))[1:])

                if np.linalg.norm(existing_trans - curr_obj_trans) < SIMILAR_POSE_DIST and \
                    (deg_diff < SIMILAR_POSE_ANGLE or diff_rot_axis[2] > SIMILAR_POSE_Z_AXIS):
                    print(f"Skipping scene {s+1} due to similar pose to existing scene.")
                    too_similar = True
                    break

            if not too_similar:
                existing_final_poses.append(np.concatenate((curr_obj_trans, curr_obj_rot)))

        if not too_similar:
            scene_id = "multiple" if not per_object else object_name[:16]
            clutter_scene_name = f"scene-{scene_id}-{gen_unique_hash()}"
            scene_output_dir = os.path.join(output_dir, clutter_scene_name)
            os.makedirs(scene_output_dir, exist_ok=True)

            vp_api = get_active_viewport()
            frame_viewport_prims(vp_api, [f"/World/objects"])
            debug_pause(world=world, step=True, num_steps=200) # Viewport needs to render for a while before capturing
            capture_viewport_to_file(vp_api, os.path.join(scene_output_dir, f"{clutter_scene_name}.png"))
            debug_pause(world=world, step=True, num_steps=200)

            usd_path = os.path.join(scene_output_dir, f"{clutter_scene_name}.usd")
            world.stage.GetRootLayer().Export(usd_path)
            print(f"Clutter scene saved to {usd_path}")

            object_poses = {}
            all_world_poses = rigid_obj.get_world_poses()
            for i in range(len(all_world_poses[0])):
                object_poses[all_object_names[i]] = {
                    "position": all_world_poses[0][i].tolist(),
                    "orientation": all_world_poses[1][i].tolist()
                }
            with open(os.path.join(scene_output_dir, f"{clutter_scene_name}.json"), 'w') as f:
                json.dump(object_poses, f, indent=4)

        stage_utils.create_new_stage()

    # Close Isaac sim
    simulation_app.close()
