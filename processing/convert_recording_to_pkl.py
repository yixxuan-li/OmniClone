import argparse
import json
import os
import pickle
import glob
import tqdm
import numpy as np

def convert_recording_to_pkl(recording_path, output_path):
    with open(recording_path, 'r') as f:
        recordings = json.load(f)
    result_dict = dict()
    root_pos = []
    root_rot = []
    joint_pos = []
    timestamps = []

    rigid_body_names = set()
    for recording in recordings:
        root_pos.append(recording['qpos'][:3])
        root_rot.append(recording['qpos'][3:7])
        joint_pos.append(recording['qpos'][7:])
        timestamps.append(recording['timestamp'])
        for key in recording['rigid_bodies'].keys():
            rigid_body_names.add(key)

    root_pos = np.array(root_pos)
    root_rot = np.array(root_rot)
    joint_pos = np.array(joint_pos)
    timestamps = np.array(timestamps)
    dts = timestamps[1:] - timestamps[:-1]
    fps = 1.0 / np.mean(dts)

    rigid_bodies_data = {key: [] for key in rigid_body_names}
    for recording in recordings:
        for key in rigid_body_names:
            if key in recording['rigid_bodies']:
                rigid_bodies_data[key].append(recording['rigid_bodies'][key])
            else:
                rigid_bodies_data[key].append([0.0, 0.0, -10000.1, 1.0, 0.0, 0.0, 0.0])
    rigid_bodies_data = {key: np.array(data) for key, data in rigid_bodies_data.items()}
        
    result_dict['root_pos'] = root_pos
    result_dict['root_rot'] = root_rot
    result_dict['joint_pos'] = joint_pos
    result_dict['fps'] = fps
    result_dict['rigid_bodies_data'] = rigid_bodies_data
    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert recording to pkl")
    parser.add_argument("--recording_path", type=str, required=True, help="Path to the recording JSON file")
    parser.add_argument("--output_name", type=str, required=True, help="Name of the output pkl file")
    parser.add_argument("--output_path", type=str, default="datas/pkls/", help="Path to the output pkl file")
    args = parser.parse_args()

    recording_paths = glob.glob(args.recording_path)
    print(f"Found {len(recording_paths)} recordings, converting to pkl")
    os.makedirs(args.output_path, exist_ok=True)
    
    results = {}
    for recording_path in tqdm.tqdm(recording_paths):
        recording_name = os.path.basename(recording_path).replace(".json", "")
        result_dict = convert_recording_to_pkl(recording_path, os.path.join(args.output_path, recording_name + ".pkl"))
        results[recording_name] = result_dict

    with open(os.path.join(args.output_path, f"{args.output_name}.pkl"), 'wb') as f:
        pickle.dump(results, f)