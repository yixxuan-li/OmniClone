from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import threading
import argparse
import time
import socket
from general_motion_retargeting.pico_client import XrClient
# from hand_retargeting import start_retargeting, HAND_RETARGET_DIMS
import numpy as np
import json
import torch



def main(args):

    # the client to receive policy inputs
    client_address = args.client_ip
    client_port = args.client_port
    
    client = XrClient()

    retarget = GMR(
        src_human="pico",
        tgt_robot=args.robot,
        actual_human_height=1.32,
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
        # num_retarget_joints, num_asset_joints, num_real_joints = HAND_RETARGET_DIMS[args.hand_type]
        # left_hand_start_index = num_retarget_joints
        # right_hand_start_index = num_retarget_joints+num_asset_joints+num_real_joints+num_retarget_joints

        # from multiprocessing import Process
        # from multiprocessing.shared_memory import SharedMemory
        # # hand_array is the mocap hands
        # hand_array_shm = SharedMemory(create=True, size=2*25*3*4)
        # hand_array = np.ndarray((2, 25, 3), dtype=np.float32, buffer=hand_array_shm.buf)
        # # hand_qpos is the retargeting results of hands
        # hand_qpos_shm = SharedMemory(create=True, size=2*(num_retarget_joints+num_asset_joints+num_real_joints)*4)
        # hand_qpos = np.ndarray(2*(num_retarget_joints+num_asset_joints+num_real_joints), dtype=np.float32, buffer=hand_qpos_shm.buf)
        
        # # if we change the hand qpos according to the pico button state, we need to change the process to monitor the pico buttin state here
        # hand_retargeting_process = Process(target=start_retargeting, args=(args.hand_type, hand_array_shm.name, hand_qpos_shm.name))
        # hand_retargeting_process.start()
    
    
    # initialize the socket for sending data to client
    policy_input_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # initialize the viewer
    # robot_type = "unitree_g1_inspire" if args.use_hand else "unitree_g1"
    robot_type = "unitree_g1"
    viewer = RobotMotionViewer(robot_type=robot_type, transparent_robot=1, motion_fps=1000)
    
    # initialize qpos array
    full_qpos = viewer.data.qpos.copy()
    
    while True:
        raw_frame = client.get_body_tracking_data()
        frame = client.get_body_tracking_human_frame(raw_frame)

        # get button state to decide whether to open or close the hands, true means close, false means open
        right_hand_state = client.get_button_state_by_name('A')
        left_hand_state = client.get_button_state_by_name('X')

        if frame is None:
            print("No body tracking data available.")
            time.sleep(0.1)
            continue

        qpos = retarget.retarget(frame)
        # set qpos
        if qpos is not None:
            qpos[2] += -0.05
            full_qpos[:3] = qpos[:3]
            full_qpos[3:7] = qpos[3:7]
            if args.use_hand:
                # full_qpos[7:hand_start_index] = qpos[7:hand_start_index]
                # full_qpos[hand_start_index+hand_joint_num:hand_start_index+hand_joint_num+second_hand_delta] = qpos[hand_start_index:]
                full_qpos[7:] = qpos[7:]
                # pass            
            else:
                full_qpos[7:] = qpos[7:]
        # initialize the send data to server_root_vel
        send_data = {}
        # print(f"frame: {frame}")
        hip_rot = frame["pelvis"][1]
        head_rot = frame["head"][1]
        send_data['hip_rot'] = hip_rot.tolist()
        send_data['head_rot'] = head_rot.tolist()
        
        # use hand retargeting
        # if args.use_hand:
        #     # mocap hands data, pico use one button to open/close hand
        #     # hand_data = server.get_latest_hand()
        #     # if hand_data is not None:
        #     #     hand_array[:, hand_open_xr, :] = hand_data
            
        #     # target hand qpos to be sent to policy client, modified according to the pico hand open/close state
        #     left_qpos = hand_qpos[left_hand_start_index:left_hand_start_index+num_asset_joints]
        #     right_qpos = hand_qpos[right_hand_start_index:right_hand_start_index+num_asset_joints]
        #     full_qpos[hand_start_index:hand_start_index+hand_joint_num] = left_qpos
        #     full_qpos[hand_start_index+hand_joint_num+second_hand_delta:hand_start_index+second_hand_delta+2*hand_joint_num] = right_qpos
        
        # get rigid body data, send to policy client
        rigid_body_data = retarget.scaled_human_data
        send_data['rigid_bodies'] = {}
        for key, item in rigid_body_data.items():
            item_concat = np.concatenate((item[0], item[1]))
            # print(f"item_concat: {item_concat}")
            valid = np.linalg.norm(item_concat[:3]) < 10000.0
            if not valid:
                continue
            # item_concat[2] -= 0.4
            send_data['rigid_bodies'][key] = item_concat.tolist()
        
        viewer.step(
            root_pos=full_qpos[:3],
            root_rot=full_qpos[3:7],
            dof_pos=full_qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            rate_limit=False,
            show_human_body_name=False,
        )
        if qpos is not None:
            send_data['qpos'] = qpos.tolist()
            reference_body_data = viewer.get_ref_body_data()
            for key, item in reference_body_data.items():
                if type(item) == torch.Tensor:
                    reference_body_data[key] = reference_body_data[key].tolist()
            send_data.update(reference_body_data)

        # only have close and open now
        right_hand_list = [[]]
        left_hand_list = [[]]
        if right_hand_state:
            right_hand_list = [[0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
        else:
            right_hand_list = [[1, 1, 1, 1, 1, 1]]

        if left_hand_state:
            left_hand_list = [[0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
        else:
            left_hand_list = [[1, 1, 1, 1, 1, 1]]

        if args.use_hand:
            # send_data['lhand_qpos'] = hand_qpos[left_hand_start_index:left_hand_start_index+num_asset_joints].tolist()
            # send_data['rhand_qpos'] = hand_qpos[right_hand_start_index:right_hand_start_index+num_asset_joints].tolist()
            send_data['lhand_qpos_real'] = left_hand_list
            send_data['rhand_qpos_real'] = right_hand_list
        send_data['timestamp'] = time.monotonic()
        # print(f"send_data: {send_data}")
        send_data = json.dumps(send_data).encode()
        policy_input_sender.sendto(send_data, (client_address, client_port))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--use_hand", action="store_true")
    parser.add_argument("--hand_type", default="inspire_hand", type=str)
    parser.add_argument("--client_ip", type=str, default="192.168.8.102")
    parser.add_argument("--client_port", type=int, default=7002)

    args = parser.parse_args()
    main(args)
    
