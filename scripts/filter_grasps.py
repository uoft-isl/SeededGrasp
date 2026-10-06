"""Keep only the successful grasps (result == 1.0) of every result .json in --input_dir.

Works on the output of standalone.py (e.g. the 6-axis test) and of get_feasible_grasps.py.
Output files keep the same name and contain gripper_id, object_id, (scene_id), pose, dofs,
stable_poses, stable_dofs and result for the passing grasps.
"""
import os
import json
import argparse

GRASP_FIELDS = ['pose', 'dofs', 'result', 'stable_poses', 'stable_dofs']

def parse_args():
    parser = argparse.ArgumentParser(description='Filter grasps based on JSON results.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing JSON result files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir to save filtered grasps.')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for js in sorted(os.listdir(args.input_dir)):
        if not js.endswith('.json'):
            continue
        try:
            with open(os.path.join(args.input_dir, js), 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"Error reading {js}. Skipping...")
            continue

        # Create a copy of the json with all passing grasps
        filtered_data = {
            'gripper_id': data.get('gripper_id', data.get('gripper')),
            'object_id': data['object_id'],
        }
        if 'scene_id' in data:
            filtered_data['scene_id'] = data['scene_id']
        for field in GRASP_FIELDS:
            filtered_data[field] = [v for v, result in zip(data[field], data['result']) if result == 1.0]

        # Save the filtered data to a new JSON file
        output_path = os.path.join(args.output_dir, js)
        with open(output_path, 'w') as out_f:
            json.dump(filtered_data, out_f, indent=4)
        print(f"Filtered data saved to {output_path}: {len(filtered_data['pose'])}/{len(data['pose'])} grasps passed.")
