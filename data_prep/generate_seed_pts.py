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

import argparse
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vlm_utils import VLM_FAMILIES, BatchedPromptV3_1, build_vlm, set_nested


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./vlm_input")
    parser.add_argument("--output_dir", type=str, default="./vlm_output")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--continue_json", action="store_true")
    parser.add_argument(
        "--vlm",
        type=str,
        default="gemini",
        choices=sorted(VLM_FAMILIES),
        help="VLM family used for seed-point prediction (default: gemini, "
        "which is the model used for all results in the paper).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit OpenRouter model id, overriding the family default "
        '(e.g. "openai/gpt-5.6-luna"). Use this to reproduce the VLM ablation.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Setup
    args = parse_args()
    model = build_vlm(args.vlm, args.model)
    print(f"Using VLM: {model.get_model_id()}")
    msg_template = BatchedPromptV3_1()

    # Read input jsons
    for j in os.listdir(args.input_dir):
        if f"seed_pts_{j}" in os.listdir(args.output_dir) and not args.continue_json:
            print(f"Input json {j} already processed, skipping...")
            continue

        output_file_path = os.path.join(args.output_dir, f"seed_pts_{j}")
        output_json = {}
        if args.continue_json and os.path.exists(output_file_path):
            with open(output_file_path, "r") as f:
                output_json = json.load(f)

        with open(os.path.join(args.input_dir, j), "r") as f:
            scene_info = json.load(f)

        # Batch input
        queue = []
        # Current method of batching will prompt same scene-object multiple times (for each gripper)
        # Leave for now b/c might want to specialize prompts to gripper later anyways
        # Only cost is extra token usage
        skipped = 0
        for gripper_id in scene_info.keys():
            for scene_id in scene_info[gripper_id].keys():
                for obj_id in scene_info[gripper_id][scene_id].keys():
                    # Skip entries already present in the output when continuing
                    if (
                        args.continue_json
                        and output_json.get(gripper_id, {}).get(scene_id, {}).get(obj_id)
                        is not None
                    ):
                        skipped += 1
                        continue
                    queue.append(
                        {
                            "gripper_id": gripper_id,
                            "scene_id": scene_id,
                            "obj_id": obj_id,
                            "obj_desc": f"{scene_info[gripper_id][scene_id][obj_id]['obj_col']} {scene_info[gripper_id][scene_id][obj_id]['obj_name']}",  # can change later
                            "img_path": scene_info[gripper_id][scene_id][obj_id]["img_path"],
                            "pcd_path": scene_info[gripper_id][scene_id][obj_id]["pcd_path"],
                        }
                    )
        if skipped:
            print(f"Skipping {skipped} already-processed entries (--continue_json).")
        queue = [queue[i : i + args.batch_size] for i in range(0, len(queue), args.batch_size)]

        # Prompt VLM
        for batch_idx, batch in enumerate(queue):
            print(f"Starting batch {batch_idx+1}/{len(queue)}")
            start_time = time.time()

            # Load img and pcds
            batch_input = []
            batch_imgs = []
            batch_pcds = []
            for batch_entry in batch:
                batch_input.append(
                    [batch_entry["obj_desc"], model.load_img(batch_entry["img_path"])]
                )
                batch_imgs.append(Image.open(batch_entry["img_path"]))
                batch_pcds.append(np.array(o3d.io.read_point_cloud(batch_entry["pcd_path"]).points))

            # Construct msg
            msg = msg_template.construct_msg(batch_input)

            # Prompt model
            response, parsed_response = None, None
            try:
                response = model.prompt(msg)
                parsed_response = msg_template.extract_json(response)
            except Exception as e:
                print("An error has occurred.", e)
                print(response)
                print(parsed_response)
                exit(1)
                continue

            print(f"Batch {batch_idx+1}/{len(queue)} took {time.time()-start_time:.2f}s")

            # Post-process results
            print("Processing results")
            try:
                for response_idx in parsed_response.keys():
                    idx = int(response_idx) - 1
                    response_entry = parsed_response[response_idx]
                    set_nested(
                        output_json,
                        [batch[idx]["gripper_id"], batch[idx]["scene_id"], batch[idx]["obj_id"]],
                        msg_template.post_process(batch_imgs[idx], batch_pcds[idx], response_entry),
                    )
            except Exception as e:
                print("An error has occurred (likely unexpected response format).", e)
                print(response)
                print(parsed_response)
                continue

            with open(output_file_path, "w") as f:
                json.dump(output_json, f, indent=4)

    print("Done")
