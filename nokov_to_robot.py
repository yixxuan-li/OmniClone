import threading
import argparse
import socket
import json
from sympy import Q
import torch
import numpy as np
import time


from nokov_server import Nokov_Server
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from hand_retargeting import start_retargeting, HAND_RETARGET_DIMS
from deploy_utils.inspire_service import InspireController
import pinocchio as pin
from pinocchio.utils import *

        
def main(args):
    '''
    This function is used to retarget the motion from Nokov to the robot and send the data to the client.
    (either using the rigid body info or the joint position)

    input:
        args: arguments
        args.robot: the robot type
        args.mocap_ip: the IP address of the mocap server
        args.client_ip: the IP address of the client
        args.client_port: the port of the client
    output: None
    '''
    
    # Check if firewall is disabled on this machine
    print("Make sure to disable firewall on both machines:")
    print("On OptiTrack computer: Disable Windows Firewall")
    print("On this computer: sudo ufw disable")

    client_address = args.client_ip
    client_port = args.client_port

    # initialize the Nokov server
    server = Nokov_Server(args.mocap_ip)


    if not server:
        print("Failed to setup Nokov server")
        exit(1)

    # initialize the retargeting system
    retarget = GMR(
            src_human="nokov",
            tgt_robot=args.robot,
            actual_human_height=1.88,
        )

    if args.use_hand:
        hand_joint_num = 0
        hand_start_index = 7 + 22
        second_hand_delta = 29 - 22
        if args.hand_type == "inspire_hand":
            hand_joint_num = 11
        elif args.hand_type == "unitree_dex3":
            hand_joint_num = 7
        hand_open_xr = [0, 4, 9, 14, 19, 24]
        num_retarget_joints, num_asset_joints, num_real_joints = HAND_RETARGET_DIMS[args.hand_type]
        left_hand_start_index = num_retarget_joints
        right_hand_start_index = num_retarget_joints+num_asset_joints+num_real_joints+num_retarget_joints

        from multiprocessing import Process
        from multiprocessing.shared_memory import SharedMemory
        hand_array_shm = SharedMemory(create=True, size=2*25*3*4)
        hand_array = np.ndarray((2, 25, 3), dtype=np.float32, buffer=hand_array_shm.buf)
        hand_qpos_shm = SharedMemory(create=True, size=2*(num_retarget_joints+num_asset_joints+num_real_joints)*4)
        hand_qpos = np.ndarray(2*(num_retarget_joints+num_asset_joints+num_real_joints), dtype=np.float32, buffer=hand_qpos_shm.buf)
        hand_retargeting_process = Process(target=start_retargeting, args=(args.hand_type, hand_array_shm.name, hand_qpos_shm.name))
        hand_retargeting_process.start()

    # initialize the socket for sending data to client
    retarget_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # initialize the viewer
    robot_type = "unitree_g1_inspire" if args.use_hand else "unitree_g1"
    viewer = RobotMotionViewer(robot_type=robot_type, transparent_robot=1, motion_fps=1000) # no limit on the motion fps

    # initialize the data recorder
    if args.record_data:
        recordings = []

    # initialize qpos array
    full_qpos = viewer.data.qpos.copy()
    try:
        last_record_time = time.monotonic()
        while True:
            # get the latest frame from the Nokov server
            frame = server.get_latest_frame()
            # print("frame: ", frame)
            # retarget the motion from Nokov to the robot
            if frame is not None:
                qpos = retarget.retarget(frame)
            else:
                qpos = None
            
            # set qpos
            if qpos is not None:
                full_qpos[:3] = qpos[:3]
                full_qpos[3:7] = qpos[3:7]
                if args.use_hand:
                    full_qpos[7:hand_start_index] = qpos[7:hand_start_index]
                    full_qpos[hand_start_index+hand_joint_num:hand_start_index+hand_joint_num+second_hand_delta] = qpos[hand_start_index:]
                else:
                    full_qpos[7:] = qpos[7:]

            # initialize the send data
            send_data = {}

            hip_rot = frame["Hips"][1]
            head_rot = frame["Head"][1]
            send_data['hip_rot'] = hip_rot.tolist()
            send_data['head_rot'] = head_rot.tolist()
            # use hand retargeting
            if args.use_hand:
                hand_data = server.get_latest_hand()
                if hand_data is not None:
                    hand_array[:, hand_open_xr, :] = hand_data
                left_qpos = hand_qpos[left_hand_start_index:left_hand_start_index+num_asset_joints]
                right_qpos = hand_qpos[right_hand_start_index:right_hand_start_index+num_asset_joints]
                full_qpos[hand_start_index:hand_start_index+hand_joint_num] = left_qpos
                full_qpos[hand_start_index+hand_joint_num+second_hand_delta:hand_start_index+second_hand_delta+2*hand_joint_num] = right_qpos
        
            # get rigid body data
            rigid_body_data = server.get_latest_rigid_body()
            send_data['rigid_bodies'] = {}
            formed_rb_data = {}
            for key, item in rigid_body_data.items():
                item = item.copy()
                item[2] *= 1.35 / 1.8 # scale the z-axis to the retargeted height
                valid = np.linalg.norm(item[:3]) < 10000.0
                if not valid:
                    continue
                
                send_data['rigid_bodies'][key] = item.tolist()
                formed_rb_data[key] = [item[:3], item[3:]]

            # visualize the motion
            viewer.step(
                root_pos=full_qpos[:3],
                root_rot=full_qpos[3:7],
                dof_pos=full_qpos[7:],
                human_point_scale=0.5,
                human_motion_data=formed_rb_data,
                show_human_body_name=True,
                rate_limit=False,
            )

            if qpos is not None:
                send_data['qpos'] = qpos.tolist()
                reference_body_data = viewer.get_ref_body_data()
                for key, item in reference_body_data.items():
                    if type(item) == torch.Tensor:
                        reference_body_data[key] = reference_body_data[key].tolist()
                send_data.update(reference_body_data)

            if args.use_hand:
                send_data['lhand_qpos'] = hand_qpos[left_hand_start_index:left_hand_start_index+num_asset_joints].tolist()
                send_data['rhand_qpos'] = hand_qpos[right_hand_start_index:right_hand_start_index+num_asset_joints].tolist()
                send_data['lhand_qpos_real'] = hand_qpos[left_hand_start_index+num_asset_joints:left_hand_start_index+num_asset_joints+num_real_joints].tolist()
                send_data['rhand_qpos_real'] = hand_qpos[right_hand_start_index+num_asset_joints:right_hand_start_index+num_asset_joints+num_real_joints].tolist()
            send_data['timestamp'] = time.monotonic()
            send_data = json.dumps(send_data).encode()
            retarget_sender.sendto(send_data, (client_address, client_port))

            if args.record_data and time.monotonic() - last_record_time > args.record_data_interval:
                recordings.append(send_data)
                last_record_time = time.monotonic()

    except KeyboardInterrupt:
        import os
        import datetime
        if args.record_data:
            os.makedirs(args.record_data_path, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, data in enumerate(recordings):
                recordings[i] = json.loads(data.decode())
            json.dump(recordings, open(os.path.join(args.record_data_path, f'recordings_{timestamp}.json'), 'w'))
            print(f"Saved recordings to {os.path.join(args.record_data_path, f'recordings_{timestamp}.json')}, length: {len(recordings)}")
        print("Keyboard interrupt. Exiting...")
        viewer.close()
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mocap_ip", type=str, default="192.168.123.101")
    parser.add_argument("--client_ip", type=str, default="127.0.0.1")
    parser.add_argument("--client_port", type=int, default=7002)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--use_hand", action="store_true")
    parser.add_argument("--hand_type", type=str, default="inspire_hand", choices=["inspire_hand", "unitree_dex3", "brainco_hand"])
    parser.add_argument("--record_data", action="store_true")
    parser.add_argument("--record_data_path", type=str, default="./recordings/")
    parser.add_argument("--record_data_interval", type=float, default=0.02)
    args = parser.parse_args()
    main(args)
    