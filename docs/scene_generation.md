# Cluttered Scene Generation and Scene-Aware Grasp Filtering

The scripts in [scripts/](../scripts) extend the MGG toolkit from single, free-floating objects to objects resting on a table. The pipeline is:

1. **6-axis test**: evaluate each object-gripper grasp file with the `axes_forces` test and keep the grasps that pass.
2. **Scene generation**: drop objects onto a ground plane and save the settled scenes.
3. **Scene filtering**: place the grasps from step 1 in every scene and reject those that are kinematically infeasible (approach from below, gripper below the table, collisions with objects, ...).
4. **Rendering**: RGB-D images of the scenes and point clouds of the grippers/objects.

All scripts must be run with Isaac Sim's `python.sh` (tested with Isaac Sim 4.2.0). See the main [README](../README.md) for the installation.

The commands below assume the MGG objects (`<objects>/<object>/<object>.usd`) and grasps (`<grasps>/<gripper>/<gripper>-<object>.json`) from the [MGG dataset](https://utdallas.box.com/v/multi-gripper-grasp-data). Use absolute paths.

## 1. 6-axis test

Run the test with the grippers in [fast_grippers](../fast_grippers), which are copies of the grippers with much higher joint velocity limits (see its [README](../fast_grippers/README.md)). For each gripper:

```Shell
./python.sh (repo directory)/standalone.py --json_dir=<grasps>/<gripper> --gripper_dir=(repo directory)/fast_grippers --objects_dir=<objects> \
    --output_dir=<data>/6_axis_grasps/<gripper> --num_w=25 --test_time=6 --test_type=axes_forces --controller=static --dof_given --headless \
    --/log/outputStreamLevel=error
./python.sh (repo directory)/scripts/filter_grasps.py --input_dir=<data>/6_axis_grasps/<gripper> --output_dir=<data>/6_axis_filtered_grasps/<gripper>
```

Besides the fields described in [grasp_data_format.md](grasp_data_format.md), the output contains `stable_poses` and `stable_dofs`: the gripper pose relative to the object and the joint values at the end of the test. The following steps use these instead of the original `pose`/`dofs`. `standalone.py --stable_poses` loads them as input.

`filter_grasps.py` keeps only the grasps with `result == 1.0`. The following steps expect the filtered files at `<data>/6_axis_filtered_grasps/<gripper>/<gripper>-<object>.json`.

## 2. Scene generation

```Shell
# Single-object scenes: up to 6 distinct resting poses per object out of 20 random drops
./python.sh scripts/gen_clutter_scene.py --objects_dir=<objects> --output_dir=<data>/scenes --per_object --num_scenes=20 --headless
# Cluttered scenes: 100 scenes with 1-5 random objects each
./python.sh scripts/gen_clutter_scene.py --objects_dir=<objects> --output_dir=<data>/scenes --num_scenes=100 --num_objects=5 --rand_obj_num --headless
```

Objects are distinct within a scene but can appear in several scenes. `--obj_list_path` restricts the objects to the names listed in a text file (one per line). Each scene is written to `<data>/scenes/scene-<object or "multiple">-<hash>/`:

- `<scene>.usd`: the full stage (objects are referenced by absolute path)
- `<scene>.json`: `{object_name: {"position": [x, y, z], "orientation": [qw, qx, qy, qz]}}` with the settled world poses
- `<scene>.png`: viewport capture

## 3. Scene filtering

```Shell
./python.sh scripts/get_feasible_grasps.py --scene_dir=<data>/scenes --grasp_dir=<data>/6_axis_filtered_grasps --object_dir=<objects> \
    --gripper_dir=(repo directory)/grippers --gripper=<gripper> --output_dir=<data>/scene_grasps/<gripper> --num_workers=20 --headless
./python.sh scripts/filter_grasps.py --input_dir=<data>/scene_grasps/<gripper> --output_dir=<data>/scene_filtered_grasps/<gripper>
```

For every object in every scene, the object's `stable_poses`/`stable_dofs` are placed in the scene, one grasp per workstation, and a grasp is rejected if

- the gripper approaches from too low an angle (elevation of the gripper seen from the object's ground projection < 0.5 rad),
- the target object moves after the gripper is spawned (> 5 mm or > 0.05 rad),
- any gripper link origin is below 2 cm, or any sampled point of the gripper base mesh is below 5 mm,
- the PhysX contact report contains a contact deeper than 1 mm involving any scene object.

Only the objects report contacts (the gripper USDs have no contact report API). Contacts between the gripper and an object are therefore detected, but contacts between the gripper and the table are not: the height checks above are what keep the gripper out of the table. Since the check does not require the gripper to be part of the contact, an object-object or object-table contact deeper than 1 mm (e.g. objects that did not fully settle in a cluttered scene) also rejects the grasp.

The thresholds are defined at the top of [get_feasible_grasps.py](../scripts/get_feasible_grasps.py). The output is `<scene>-<gripper>-<object>-feasible_grasps.json` with `object_id`, `gripper_id`, `scene_id`, `pose`, `dofs`, `stable_poses`, `stable_dofs` and `result` (1.0 feasible, 0.0 rejected). Existing output files are skipped, so interrupted runs can be resumed.

## 4. Rendering

```Shell
# 5 RGB images + point clouds per scene, written into each scene folder; camera poses are added to <scene>.json
./python.sh scripts/get_scene_rgbd.py --scene_dir=<data>/scenes
# Surface point clouds (full, downsampled, downsampled + noise) of all grippers and objects
./python.sh scripts/get_point_cloud.py --gripper_dir=(repo directory)/grippers --object_dir=<objects> --output_dir=<data>/point_clouds --headless
```

Gripper point clouds are sampled with every joint at the middle of its limits and are expressed in the world frame of the spawned gripper, which sits 1 m above the origin, rotated according to its `EF_axis`.

## Visualization

```Shell
./python.sh scripts/visualize_grasps.py --scene_dir=<data>/scenes --grasp_dir=<data>/scene_grasps/<gripper> --object_dir=<objects> \
    --gripper_dir=(repo directory)/grippers --gripper=<gripper> --output_dir=<data>/screenshots/<gripper> --num_screenshots=10
```

For every scene-object pair, screenshots of up to `--num_screenshots` feasible and infeasible grasps are saved as `<scene>-<gripper>-<object>_<idx>_<result>.png`.
