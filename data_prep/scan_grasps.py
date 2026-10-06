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
scan_grasps.py - Scan a directory of robot grasp data and produce a VLM input JSON.

Usage:
    python scan_grasps.py --scenes_dir <dir> --output_json <file>
                          [--data_dir <dir>] [--min-grasps N] [--annotations <csv>]

Arguments:
    --data_dir      Directory containing per-gripper subdirectories of feasible_grasps.json
                    files. Optional: if omitted, every object found in each scene's JSON is
                    included for every known gripper, with no feasible-grasp-count
                    thresholding (--min-grasps is ignored in that case).
    --scenes_dir    Directory containing per-scene image/pcd data.
    --output_json   Output JSON file path.
    --min-grasps    Minimum number of grasps to include (default: 10). Only used when
                    --data_dir is provided.
    --annotations   Optional CSV with object metadata (columns: object_id, object_name,
                    object_colour, object_description). When provided, each entry will
                    include 'obj_name' and 'obj_col' fields matching test.json format.

Output format:
    {
        "<gripper>": {
            "<scene_id>": {
                "<object_id>": {
                    "obj_name":  "<object_name>",   # only when --annotations provided
                    "obj_col":   "<object_colour>", # only when --annotations provided
                    "img_path": "<scenes_dir>/<scene_id>/camera_bev_rgb.png",
                    "pcd_path": "<scenes_dir>/<scene_id>/camera_bev_depth.pcd"
                }
            }
        }
    }
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

KNOWN_GRIPPERS = ["Allegro", "franka_panda", "robotiq_3finger"]


def load_annotations(csv_path: str) -> dict:
    """Load object metadata from a CSV into a dict keyed by object_id.

    Expected columns: object_id, object_name, object_colour, object_description
    Returns: {object_id: {"obj_name": ..., "obj_col": ..., "obj_desc": ...}}
    """
    annotations = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = row.get("object_id", "").strip()
            if oid:
                annotations[oid] = {
                    "obj_name": row.get("object_name", "").strip(),
                    "obj_col": row.get("object_colour", "").strip(),
                }
    print(f"Loaded annotations for {len(annotations)} objects from {csv_path}")
    return annotations


def count_grasps(filepath: str) -> int:
    """Count the number of feasible grasps in a grasp JSON file.

    Each file is a dict with a 'pose' list (one entry per grasp).
    Falls back to 'result' if 'pose' is missing. Returns 0 on any error.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        # 'pose' is the primary list of grasps; fall back to 'result'
        if isinstance(data, dict):
            poses = data.get("pose") or data.get("result") or []
            return len(poses)
        elif isinstance(data, list):
            return len(data)
    except Exception as e:
        print(f"  Warning: could not read {filepath}: {e}", file=sys.stderr)
    return 0


def parse_filename(filename: str):
    """Parse a feasible_grasps filename into (scene_id, gripper_id, object_id).

    Expected format: {scene_id}-{gripper_id}-{object_id}-feasible_grasps.json
    Example: scene-multiple-0af9f30fcb159cdc-Allegro-2_of_Jenga_Classic_Game-feasible_grasps.json

    scene_id always starts with 'scene-multiple-' followed by a 16-char hex hash.
    """
    suffix = "-feasible_grasps.json"
    if not filename.endswith(suffix):
        return None
    stem = filename[: -len(suffix)]
    # scene_id is everything up through the 16-char hex part (fixed prefix length)
    # Pattern: scene-multiple-<16hex>-<gripper>-<object>
    parts = stem.split("-")
    # 'scene-multiple-<hex>' is always 3 dash-separated parts at the start
    # We know scene_id = parts[0]+'-'+parts[1]+'-'+parts[2]
    if len(parts) < 4 or parts[0] != "scene" or parts[1] != "multiple":
        return None
    scene_id = "-".join(parts[:3])  # scene-multiple-<hex>
    rest = "-".join(parts[3:])  # <gripper>-<object>
    for gripper in KNOWN_GRIPPERS:
        prefix = gripper + "-"
        if rest.startswith(prefix):
            object_id = rest[len(prefix) :]
            return scene_id, gripper, object_id
    return None


def load_scene_positions(scenes_dir: str, scene_id: str) -> dict:
    """Load object XY positions from the per-scene JSON file.

    Returns: {object_id: (x, y)} or {} if the file can't be read.
    The scene JSON lives at: <scenes_dir>/<scene_id>/<scene_id>.json
    Each object entry has a 'position' list [x, y, z]. Camera entries
    (e.g. "camera_bev", "camera_north") have the same shape but aren't
    graspable objects, so they're excluded.
    """
    json_path = os.path.join(scenes_dir, scene_id, f"{scene_id}.json")
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not read scene JSON {json_path}: {e}", file=sys.stderr)
        return {}
    positions = {}
    for key, val in data.items():
        if key.startswith("camera"):
            continue
        if isinstance(val, dict) and "position" in val:
            pos = val["position"]
            if len(pos) >= 2:
                positions[key] = (pos[0], pos[1])
    return positions


def scan_directory(
    data_dir: str, scenes_dir: str, min_grasps: int, annotations: dict = None, xy_limit: float = 0.4
) -> dict:
    """Scan data_dir and return a nested dict filtered by min_grasps.

    If *annotations* is provided (dict keyed by object_id), each entry will
    include 'obj_name' and 'obj_col' fields.

    Objects whose XY position (read from the per-scene JSON) falls outside
    [-xy_limit, xy_limit] on either axis are excluded.
    """
    result = defaultdict(lambda: defaultdict(dict))
    total_files = 0
    included = 0
    xy_skipped = 0
    # Cache positions per scene so we only read each JSON once
    _scene_positions: dict = {}

    gripper_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not gripper_dirs:
        print(f"No subdirectories found in {data_dir}", file=sys.stderr)
        return {}

    for gripper_dir in sorted(gripper_dirs):
        gripper_path = os.path.join(data_dir, gripper_dir)
        files = [f for f in os.listdir(gripper_path) if f.endswith("-feasible_grasps.json")]
        print(f"Scanning {gripper_dir}: {len(files)} files...")

        for filename in sorted(files):
            total_files += 1
            parsed = parse_filename(filename)
            if parsed is None:
                print(f"  Skipping unrecognized file: {filename}", file=sys.stderr)
                continue

            scene_id, gripper_id, object_id = parsed
            filepath = os.path.join(gripper_path, filename)
            n_grasps = count_grasps(filepath)

            if n_grasps >= min_grasps:
                # ---- XY position filter ----
                if scene_id not in _scene_positions:
                    _scene_positions[scene_id] = load_scene_positions(scenes_dir, scene_id)
                pos = _scene_positions[scene_id].get(object_id)
                if pos is not None:
                    x, y = pos
                    if abs(x) > xy_limit or abs(y) > xy_limit:
                        xy_skipped += 1
                        continue
                else:
                    print(
                        f"  [WARN] No position found for object '{object_id}' in scene '{scene_id}'",
                        file=sys.stderr,
                    )

                img_path = os.path.join(scenes_dir, scene_id, "camera_bev_rgb.png")
                pcd_path = os.path.join(scenes_dir, scene_id, "camera_bev_depth.pcd")
                entry: dict = {}
                # Inject annotation fields first (matching test.json field order)
                if annotations is not None:
                    ann = annotations.get(object_id, {})
                    if ann:
                        entry["obj_name"] = ann["obj_name"]
                        entry["obj_col"] = ann["obj_col"]
                    else:
                        # Object not in CSV — leave fields out rather than crash
                        print(f"  [WARN] No annotation for object: {object_id}", file=sys.stderr)
                entry["img_path"] = img_path
                entry["pcd_path"] = pcd_path
                result[gripper_id][scene_id][object_id] = entry
                included += 1

    print(f"\nScanned {total_files} files total.")
    print(f"Included {included} (gripper, scene, object) combos with >= {min_grasps} grasps.")
    if xy_skipped:
        print(f"Excluded {xy_skipped} combos where object XY position exceeded +/-{xy_limit} m.")
    return result


def scan_scenes_no_threshold(
    scenes_dir: str, annotations: dict = None, xy_limit: float = 0.4
) -> dict:
    """Build the same nested result as scan_directory, but without any
    feasible-grasp-count thresholding (no --data_dir available to check
    against). Every object found in each scene's JSON is included for every
    gripper in KNOWN_GRIPPERS, subject only to the XY position filter.
    """
    result = defaultdict(lambda: defaultdict(dict))
    included = 0
    xy_skipped = 0

    scene_ids = sorted(
        d for d in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d))
    )
    print(f"Scanning {len(scene_ids)} scenes in {scenes_dir} (no feasible-grasp thresholding)...")

    for scene_id in scene_ids:
        positions = load_scene_positions(scenes_dir, scene_id)
        img_path = os.path.join(scenes_dir, scene_id, "camera_bev_rgb.png")
        pcd_path = os.path.join(scenes_dir, scene_id, "camera_bev_depth.pcd")

        for object_id, (x, y) in positions.items():
            if abs(x) > xy_limit or abs(y) > xy_limit:
                xy_skipped += 1
                continue

            entry: dict = {}
            if annotations is not None:
                ann = annotations.get(object_id, {})
                if ann:
                    entry["obj_name"] = ann["obj_name"]
                    entry["obj_col"] = ann["obj_col"]
                else:
                    print(f"  [WARN] No annotation for object: {object_id}", file=sys.stderr)
            entry["img_path"] = img_path
            entry["pcd_path"] = pcd_path

            for gripper_id in KNOWN_GRIPPERS:
                result[gripper_id][scene_id][object_id] = dict(entry)
            included += 1

    print(f"\nScanned {len(scene_ids)} scenes total.")
    print(f"Included {included} (scene, object) combos x {len(KNOWN_GRIPPERS)} grippers.")
    if xy_skipped:
        print(f"Excluded {xy_skipped} objects where XY position exceeded +/-{xy_limit} m.")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Scan robot grasp data and produce a VLM input JSON."
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Directory with per-gripper subdirs of grasp JSON files. Optional: if omitted, "
        "every object in --scenes_dir is included for every known gripper with no "
        "feasible-grasp-count thresholding.",
    )
    parser.add_argument("--scenes_dir", help="Directory with per-scene image/pcd data")
    parser.add_argument("--output_json", help="Output JSON file path")
    parser.add_argument(
        "--min-grasps",
        type=int,
        default=10,
        help="Minimum number of grasps to include (default: 10)",
    )
    parser.add_argument(
        "--xy-limit",
        type=float,
        default=0.4,
        metavar="M",
        help="Exclude objects whose X or Y position (metres) exceeds this absolute value "
        "(default: 0.4). Set to a large value to disable.",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        metavar="CSV",
        help="Optional CSV with object metadata (object_id, object_name, object_colour)."
        " When provided, adds obj_name and obj_col to each output entry.",
    )
    args = parser.parse_args()

    if args.data_dir is not None and not os.path.isdir(args.data_dir):
        print(f"Error: data_dir does not exist: {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.scenes_dir):
        print(f"Error: scenes_dir does not exist: {args.scenes_dir}", file=sys.stderr)
        sys.exit(1)

    annotations = None
    if args.annotations:
        if not os.path.isfile(args.annotations):
            print(f"Error: annotations file not found: {args.annotations}", file=sys.stderr)
            sys.exit(1)
        annotations = load_annotations(args.annotations)

    if args.data_dir is not None:
        result = scan_directory(
            args.data_dir, args.scenes_dir, args.min_grasps, annotations, xy_limit=args.xy_limit
        )
    else:
        result = scan_scenes_no_threshold(args.scenes_dir, annotations, xy_limit=args.xy_limit)

    # Convert defaultdicts to plain dicts for JSON serialization
    output = {
        gripper: {scene: dict(objects) for scene, objects in scenes.items()}
        for gripper, scenes in result.items()
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\nOutput written to: {args.output_json}")


if __name__ == "__main__":
    main()
