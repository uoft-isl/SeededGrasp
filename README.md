# SeededGrasp

This repository is the codebase for:

**SeededGrasp: Language-Guided Grasping in Complex Scenes with Multiple Embodiments**
Yang Xu, Gurpreet Singh Mukker, Raymond Wang, Jasper Gerigk, Maria Attarian, Igor Gilitschenski

[Project page](https://uoft-isl.github.io/seeded-grasp/) · [Paper (arXiv:2607.20207)](https://arxiv.org/abs/2607.20207)

SeededGrasp generates multi-embodiment grasps in cluttered tabletop scenes. A
vision-language model (VLM) first predicts a language-conditioned 3D "seed
point" on the target object; a lightweight flow-matching model then generates
the full grasp (pose + joint angles) for a given end-effector, conditioned on
that seed point and the local scene geometry. Decoupling the two stages means
adding a new gripper only requires retraining the small grasp-generation
model, not the VLM.

## Repository layout

This repository contains the grasp-generation model: preprocessing, training,
inference, and the VLM seed-point stage. 

The main branch contains the code for our model and inference pipeline. 
The data_gen_pipeline branch contains the code for our data generation pipeline.
You can additionally find the dataset we used on [HuggingFace](https://huggingface.co/datasets/yangxu0/SeededGrasp).

```
configs/            Hydra configs (dataset / model / train groups)
models/             Model definitions (flow-matching transformer, GNN encoders, MLPs)
utils/              Shared math, gripper, normalization, and general-purpose helpers
utils_data/         Dataset / dataloader implementations and augmentation
training/           Training entrypoints
inference/          Grasp generation entrypoints
visualization/      Plotly-based visualization scripts
data_prep/          Dataset preprocessing and VLM seed-point generation scripts
```

## Installation

Requires Python 3.10+ and a CUDA-capable GPU (tested on an RTX 3090).

```bash
conda create -n seededgrasp python=3.10
conda activate seededgrasp
pip install -r requirements.txt
```

The VLM-seeded pipeline (`data_prep/generate_seed_pts.py`) additionally needs
API access to a hosted VLM via [OpenRouter](https://openrouter.ai/). Create a
`.env` file at the repo root:

```
OPENROUTER_API_KEY=<your-key>
```

(`vlm_utils.py` loads this automatically via `python-dotenv`.)

## Data

Datasets are not included in this repository. `configs/dataset/rgbd_scene.yaml`'s
`dataset_basedir` (`./data/rgbd_final` by default) points at where cluttered,
multi-object tabletop scene data is expected on disk.

That directory is expected to contain a `hand_models.pt` (a dict of
per-gripper `HandModel` objects, see `utils/gripper_utils.py`) alongside the
dataset-specific point cloud / grasp files each dataloader reads (see
`utils_data/rgbd_scene_dataset.py`). Point your own data at these paths,
either by populating `./data/...` directly or by overriding
`dataset.dataset_basedir=<path>` on the command line.

## Quickstart: train, generate, visualize

These commands use the default config (`configs/config.yaml`): the
`flow_trans` model trained on the `rgbd_scene` dataset (cluttered,
multi-object tabletop scenes).

**1. Preprocess the dataset**

```bash
python data_prep/create_gnn_dataset_rgbd_scene.py
```

Builds the per-scene point cloud / adjacency tensors and the train/val grasp
index that the dataloader reads.

**2. Train the model**

```bash
python training/train_flow.py
```

Checkpoints and TensorBoard logs are written under
`./logs_train_flow_trans/<exp_name>_<timestamp>/`. For multi-GPU training,
use the DDP variant instead (better memory balance than `DataParallel`):

```bash
torchrun --nproc_per_node=2 training/train_flow_ddp.py
```

Override any Hydra field from the command line, e.g. to change the number of
epochs or batch size:

```bash
python training/train_flow.py train.epochs=100 train.batch_size=64
```

**3. Generate grasps**

```bash
python inference/generate_grasps_flow.py
```

Loads the most recent checkpoint under `train.log_basedir` matching
`train.exp_name`, generates grasps for the validation split, and (with the
default `model.plot_grasps=True`) opens an inline Plotly figure per sample.

**4. Visualize**

```bash
python visualization/visualize_scene_grasps.py --grasp_file <path_to_grasp_json>
python visualization/visualize_rgbd_dataset.py
```

## VLM-seeded grasp generation

Instead of generating grasps for the validation split's ground-truth seed
points (`inference/generate_grasps_flow.py`), you can drive generation from
language-guided seed points predicted by a VLM:

**1. Build the VLM input**

```bash
python data_prep/scan_grasps.py --scenes_dir <dir> --output_json vlm_input/scenes.json
```

Scans per-scene image/point-cloud data (optionally cross-referenced against
per-gripper feasible-grasp files) into the JSON schema the next step expects.

**2. Predict seed points**

```bash
python data_prep/generate_seed_pts.py --input_dir vlm_input --output_dir vlm_output
```

Prompts a hosted VLM (see the OpenRouter API key setup above) for a
normalized pixel coordinate per (gripper, scene, object), then projects it
onto the scene's point cloud to get a 3D seed point.

Defaults to Gemini 3 Flash, which is used for all results in the paper. Use
`--vlm {gemini,gpt,qwen,claude}` to switch family, or `--model` to pin an
explicit OpenRouter model id (e.g. `--model openai/gpt-5.6-luna`).

**3. Generate and inspect grasps**

```bash
python inference/generate_grasps_from_seed_pts.py \
    --seed-pts vlm_output/seed_pts_scenes.json \
    --output-dir generated_grasps/
python visualization/visualize_grasps_one_by_one.py --grasps generated_grasps/<gripper>_predicted_grasps.json
```

Generation integrates the flow with forward Euler over `--time-steps` steps
(default 5, i.e. step size 0.2) and applies classifier-free guidance with
weight `--cfg-weight` (defaults to `model.w` in the Hydra config, 1.1; set to
1.0 to disable guidance).

## Citing this work

```bibtex
@misc{xu2026seededgrasplanguageguidedgraspingcomplex,
      title={SeededGrasp: Language-Guided Grasping in Complex Scenes with Multiple Embodiments},
      author={Yang Xu and Gurpreet Singh Mukker and Raymond Wang and Jasper Gerigk and Maria Attarian and Igor Gilitschenski},
      year={2026},
      eprint={2607.20207},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.20207}
}
```

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited
Copyright 2026 Toronto Intelligent Systems Lab

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
