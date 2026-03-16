import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())


# from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import json

# def add_visual_capsule(scene, point1, point2, radius, rgba):
#     """Adds one capsule to an mjvScene."""
#     if scene.ngeom >= scene.maxgeom:
#         return
#     scene.ngeom += 1  # increment ngeom
#     # initialise a new capsule, add it to the scene using mjv_makeConnector
#     mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
#                         mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
#                         np.zeros(3), np.zeros(9), rgba.astype(np.float32))
#     mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
#                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
#                             point1[0], point1[1], point1[2],
#                             point2[0], point2[1], point2[2])

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif chr(keycode) == "T":
        print("next")
        motion_id += 1
        curr_motion_key = motion_data_keys[motion_id]
        print(curr_motion_key)
    else:
        print("not mapped", chr(keycode))
    
    
        
def main() -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    device = torch.device("cpu")
    humanoid_xml = 'GMR/assets/unitree_g1/g1_mocap_29dof.xml'
    
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False
    
    # motion_file = f"source/latentctrl/latentctrl/motions/g1/categories/twist.pkl"
    # motion_data = joblib.load(motion_file)
    # motion_data_keys = list(motion_data.keys())
    
    data_path = "/home/lyx/0108/episode_0067/data.json"
    data_file = json.load(open(data_path, 'r'))
    data_json = data_file['data']
    

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    
    
    RECORDING = False
    
    state_idx = 0

    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        while viewer.is_running():
            step_start = time.time()
            # curr_motion_key = motion_data_keys[motion_id]
            # curr_motion = motion_data[curr_motion_key]
            # curr_time = int(time_step/dt) % curr_motion['joint_pos'].shape[0]
            # mj_data.qpos[:3] = curr_motion['root_pos'][curr_time]  #(3, )
            # mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time] #(4, )
            # mj_data.qpos[7:] = curr_motion['joint_pos'][curr_time]
            # mj_data.qpos[7 + 15] = np.pi/2  # fix arm pose

            # raw_states = data_json[state_idx]['states']
            # mj_data.qpos[:3] = np.array([0, 0, 0.68])
            # mj_data.qpos[3:7] = np.array(raw_states['imu']['quat'])
            # # mj_data.qpos[3:7] = np.array([1, 0, 0, 0])
            # mj_data.qpos[7:13] = np.array(raw_states['left_leg']['qpos'])
            # mj_data.qpos[13:19] = np.array(raw_states['right_leg']['qpos'])
            # mj_data.qpos[19:22] = np.array(raw_states['waist']['qpos'])
            # mj_data.qpos[22:29] = np.array(raw_states['left_arm']['qpos'])
            # mj_data.qpos[29:36] = np.array(raw_states['right_arm']['qpos'])


            raw_states = data_json[state_idx]['task_obs']
            mj_data.qpos[:3] = np.array([0, 0, 0.68])
            # mj_data.qpos[3:7] = np.array(raw_states['mocap_body_quat'][0][0])
            # mj_data.qpos[3:7] = np.array([1, 0, 0, 0])
            mj_data.qpos[7:7+29] = np.array(raw_states['mocap_qpos'][0+7:0+7+29])


            state_idx += 1

            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt


            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
