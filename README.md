# MultiGripperGrasp Toolkit: 6-Axis Filtering and Cluttered Scenes

This repository is a fork of the [MultiGripperGrasp (MGG) Toolkit](https://github.com/IRVLUTD/isaac_sim_grasping) developed by the Intelligent Robotics and Vision Lab (IRVL) at the University of Texas at Dallas. The original grasp evaluation pipeline has been retained, and has been extended with the following modifications:

- **6-axis grasp evaluation**: the final gripper pose and joint values of each test are recorded (`stable_poses`, `stable_dofs`) and can be used as input for subsequent evaluations. A set of gripper assets with increased joint velocity limits is provided in [fast_grippers](fast_grippers) for this test.
- **Cluttered scene generation and scene-aware grasp filtering**: a pipeline that generates table-top scenes, filters grasps for feasibility within these scenes, and renders RGB-D observations and point clouds.

This README describes the extended functionality. For the original toolkit, i.e. the grasp evaluation simulation in `standalone.py`, its parameters, controllers and tests, and the import of new grippers and objects, please refer to the [original repository](https://github.com/IRVLUTD/isaac_sim_grasping) and its [documentation](https://github.com/IRVLUTD/isaac_sim_grasping#documentation).

### Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Pipeline](#pipeline)
  - [1. 6-axis test](#1-6-axis-test)
  - [2. Scene generation](#2-scene-generation)
  - [3. Scene filtering](#3-scene-filtering)
  - [4. Rendering](#4-rendering)
  - [Visualization](#visualization)
- [Changes to the original toolkit](#changes-to-the-original-toolkit)
- [Citation](#citation)
- [License](#license)

## Installation
This repository was tested using Isaac Sim 4.2.0 on Ubuntu.

1. Clone the repository.
2. Install Isaac Sim 4.2.0 following the instructions in the [Isaac Sim documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) and navigate to its installation directory, which contains `python.sh`.
3. Install the additional libraries in the Isaac Sim Python environment:
   ```Shell
   ./python.sh -m pip install tqdm plyfile open3d
   ```

All scripts are Isaac Sim standalones and must be run with Isaac Sim's `python.sh`. Deactivate any active conda environment beforehand, and use absolute paths for all directories.

## Data
The pipeline uses the objects (`<objects>/<object>/<object>.usd`) and grasps (`<grasps>/<gripper>/<gripper>-<object>.json`) of the MGG dataset, which can be downloaded from the [MGG shared folder](https://utdallas.box.com/v/multi-gripper-grasp-data). The grippers are included in this repository ([grippers](grippers) and [fast_grippers](fast_grippers)). In the commands below, `<data>` denotes the output directory of the pipeline.

## Pipeline
The pipeline consists of four steps. Each step reads the output of the previous one:

1. **6-axis test**: evaluate the grasps of each object-gripper pair with forces along the +/- x, y and z axes and keep the grasps that pass.
2. **Scene generation**: drop objects onto a ground plane and save the settled single-object or cluttered scenes.
3. **Scene filtering**: place the grasps from step 1 in every scene and reject those that are infeasible.
4. **Rendering**: render RGB-D images of the scenes and sample point clouds of the grippers and objects.

The commands below cover typical usage. Filtering criteria, output formats and further options are described in [docs/scene_generation.md](docs/scene_generation.md).

### 1. 6-axis test
For each gripper, evaluate the grasps with the `axes_forces` test using the [fast_grippers](fast_grippers) assets, then keep the successful grasps:

```Shell
./python.sh (repo directory)/standalone.py --json_dir=<grasps>/<gripper> --gripper_dir=(repo directory)/fast_grippers --objects_dir=<objects> \
    --output_dir=<data>/6_axis_grasps/<gripper> --num_w=25 --test_time=6 --test_type=axes_forces --controller=static --dof_given --headless \
    --/log/outputStreamLevel=error
./python.sh (repo directory)/scripts/filter_grasps.py --input_dir=<data>/6_axis_grasps/<gripper> --output_dir=<data>/6_axis_filtered_grasps/<gripper>
```

### 2. Scene generation
Generate single-object scenes (up to 6 distinct resting poses per object out of `--num_scenes` random drops) or cluttered scenes (up to `--num_objects` distinct objects per scene):

```Shell
# Single-object scenes
./python.sh (repo directory)/scripts/gen_clutter_scene.py --objects_dir=<objects> --output_dir=<data>/scenes --per_object --num_scenes=20 --headless
# Cluttered scenes
./python.sh (repo directory)/scripts/gen_clutter_scene.py --objects_dir=<objects> --output_dir=<data>/scenes --num_scenes=100 --num_objects=5 --rand_obj_num --headless
```

Each scene is saved to its own folder in `<data>/scenes`, containing the stage (`.usd`), the settled object poses (`.json`) and a screenshot (`.png`). Use `--obj_list_path` to restrict the scenes to the objects listed in a text file.

### 3. Scene filtering
For each gripper, place the 6-axis filtered grasps in every scene, evaluate them in parallel over `--num_workers` workstations, and keep the feasible grasps:

```Shell
./python.sh (repo directory)/scripts/get_feasible_grasps.py --scene_dir=<data>/scenes --grasp_dir=<data>/6_axis_filtered_grasps --object_dir=<objects> \
    --gripper_dir=(repo directory)/grippers --gripper=<gripper> --output_dir=<data>/scene_grasps/<gripper> --num_workers=20 --headless
./python.sh (repo directory)/scripts/filter_grasps.py --input_dir=<data>/scene_grasps/<gripper> --output_dir=<data>/scene_filtered_grasps/<gripper>
```

A grasp is rejected if the gripper approaches from too low an angle, displaces the target object, reaches below the table, or collides with a scene object. Existing output files are skipped, so interrupted runs can be resumed.

### 4. Rendering
```Shell
# RGB images and point clouds from five fixed cameras, saved into each scene folder
./python.sh (repo directory)/scripts/get_scene_rgbd.py --scene_dir=<data>/scenes
# Surface point clouds of all grippers and objects
./python.sh (repo directory)/scripts/get_point_cloud.py --gripper_dir=(repo directory)/grippers --object_dir=<objects> --output_dir=<data>/point_clouds --headless
```

### Visualization
Save screenshots of feasible and infeasible grasps (output of step 3) for every scene-object pair:

```Shell
./python.sh (repo directory)/scripts/visualize_grasps.py --scene_dir=<data>/scenes --grasp_dir=<data>/scene_grasps/<gripper> --object_dir=<objects> \
    --gripper_dir=(repo directory)/grippers --gripper=<gripper> --output_dir=<data>/screenshots/<gripper> --num_screenshots=10
```

## Changes to the original toolkit
In addition to the scripts in [scripts](scripts), the following changes were made to the original simulation:

- `standalone.py` reads grasp files from `--json_dir` and from its direct subfolders (the folder structure is kept in `--output_dir`), and skips empty grasp files.
- The output files contain `stable_poses` and `stable_dofs`, the gripper pose relative to the object and the joint values at the end of each test. The new `--stable_poses` flag uses them as the input grasps of a new run ([grasp data format](docs/grasp_data_format.md)).
- The `axes_forces` test holds the object in place during the setup phase, and continuous collision detection is enabled.

## Citation
If you use this repository, please cite our work:

    @misc{xu2026seededgrasplanguageguidedgraspingcomplex,
          title={SeededGrasp: Language-Guided Grasping in Complex Scenes with Multiple Embodiments},
          author={Yang Xu and Gurpreet Singh Mukker and Raymond Wang and Jasper Gerigk and Maria Attarian and Igor Gilitschenski},
          year={2026},
          eprint={2607.20207},
          archivePrefix={arXiv},
          primaryClass={cs.RO},
          url={https://arxiv.org/abs/2607.20207}
    }

This repository builds on the MultiGripperGrasp Toolkit, which was featured in the following paper at IROS 2024:

**MultiGripperGrasp: A Dataset for Robotic Grasping from Parallel Jaw Grippers to Dexterous Hands**

Luis Felipe Casas, Ninad Khargonkar, Balakrishnan Prabhakaran, Yu Xiang

[[paper](https://arxiv.org/pdf/2403.09841.pdf)] [[video](https://www.youtube.com/watch?v=pm1K6wbc830)] [[arXiv](https://arxiv.org/abs/2403.09841)] [[project site](https://irvlutd.github.io/MultiGripperGrasp)] [[dataset folder](https://utdallas.box.com/v/multi-gripper-grasp-data)]

Please also cite the original work:

    @misc{casas2024multigrippergraspdatasetroboticgrasping,
      title={MultiGripperGrasp: A Dataset for Robotic Grasping from Parallel Jaw Grippers to Dexterous Hands}, 
      author={Luis Felipe Casas and Ninad Khargonkar and Balakrishnan Prabhakaran and Yu Xiang},
      year={2024},
      eprint={2403.09841},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2403.09841}, 
     }

## License
This repository is released under the [GNU General Public License v3.0](LICENSE), as is the original MultiGripperGrasp Toolkit.
