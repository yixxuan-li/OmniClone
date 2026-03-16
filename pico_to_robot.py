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
    
    
    # initialize the socket for sending data to client
    policy_input_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # initialize the viewer
    # robot_type = "unitree_g1_inspire" if args.use_hand else "unitree_g1"
    robot_type = args.robot
    viewer = RobotMotionViewer(robot_type=robot_type, transparent_robot=1, motion_fps=1000)

    # initialize the data recorder
    if args.record_data:
        recordings = []

    # initialize qpos array
    full_qpos = viewer.data.qpos.copy()
    try:
        last_record_time = time.monotonic()
        while True:
            raw_frame = client.get_body_tracking_data()
            frame = client.get_body_tracking_human_frame(raw_frame)

            if frame is None:
                print("No body tracking data available.")
                time.sleep(0.1)
                continue

            qpos = retarget.retarget(frame)
            # set qpos
            if qpos is not None:
                qpos[2] += -0.03
                full_qpos[:3] = qpos[:3]
                full_qpos[3:7] = qpos[3:7]
                if args.use_hand:
                    # full_qpos[7:hand_start_index] = qpos[7:hand_start_index]
                    # full_qpos[hand_start_index+hand_joint_num:hand_start_index+hand_joint_num+second_hand_delta] = qpos[hand_start_index:]
                    pass            
                else:
                    full_qpos[7:] = qpos[7:]
            # initialize the send data to server_root_vel
            send_data = {}

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

            send_data['timestamp'] = time.monotonic()
            # print(f"send_data: {send_data}")
            send_data = json.dumps(send_data).encode()
            policy_input_sender.sendto(send_data, (client_address, client_port))

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
    parser.add_argument("--robot", type=str, default="unitree_g1", help="unitree_g1, unitree_g1_inspire, unitree_h1_2")
    parser.add_argument("--use_hand", action="store_true")
    parser.add_argument("--client_ip", type=str, default="192.168.8.102")
    parser.add_argument("--client_port", type=int, default=7002)
    parser.add_argument("--record_data", action="store_true")
    parser.add_argument("--record_data_path", type=str, default="./recordings/")
    parser.add_argument("--record_data_interval", type=float, default=0.02)
    args = parser.parse_args()
    main(args)
    
