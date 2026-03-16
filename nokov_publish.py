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
# from hand_retargeting import start_retargeting, HAND_RETARGET_DIMS
# from deploy_utils.inspire_service import InspireController
# import pinocchio as pin
# from pinocchio.utils import *

class Robot_FK:
    '''
    This class is used to compute the forward kinematics of the robot.
    '''
    def __init__(self, robot_type):
        self.robot_type = robot_type

        # load the robot model
        self.model = pin.buildModelFromUrdf(f"./GMR/assets/{robot_type}/robot.urdf")
        self.fk_data = self.model.createData()

    def compute_fk(self, dof_pos):
        '''
        This function is used to compute the forward kinematics of the robot.
        input: dof_pos: num_dof
        output: computed_fk, return the computed forward kinematics of the robot, using a dictionary
                    computed_fk['body_names']: list of body names, num_body
                    computed_fk['body_pos']: list of body positions, num_body x 3
                    computed_fk['body_rot']: list of body rotations, num_body x 4
        '''
        pin.forwardKinematics(self.model, self.fk_data, dof_pos)
        pin.updateFramePlacements(self.model, self.fk_data)

        computed_fk = {}
        computed_fk['body_names'] = []
        computed_fk['body_pos'] = []
        computed_fk['body_rot'] = []

        for i, joint in enumerate(self.model.names):
            M = self.fk_data.oMi[i]

            computed_fk['body_names'].append(joint)
            computed_fk['body_pos'].append(M.translation.T)
            computed_fk['body_rot'].append(M.rotation)

        return computed_fk
        
    

def main(args):
    '''
    This function is used to retarget the motion from Nokov to the robot and send the data to the client.
    (either using the rigid body info or the joint position)

    input:
        args: arguments
        args.use_rb: whether to use the rigid body info
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

    # initialize the socket for sending data to client
    data_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # initialize the viewer
    viewer = RobotMotionViewer(robot_type="unitree_g1", transparent_robot=1, motion_fps=1000, camera_follow=False) # no limit on the motion fps

    # initialize the data recorder
    if args.record_data:
        recordings = []

    # initialize qpos array
    full_qpos = viewer.data.qpos.copy()
    try:
        last_record_time = time.monotonic()
        while True:
            # initialize the send data
            send_data = {}

            # get rigid body data
            rigid_body_data = server.get_latest_rigid_body()
            send_data['rigid_bodies'] = {}
            formed_rb_data = {}
            for key, item in rigid_body_data.items():
                item = item.copy()
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
            send_data = json.dumps(send_data).encode()
            data_sender.sendto(send_data, (client_address, client_port))

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
    parser.add_argument("--client_ip", type=str, default="192.168.123.102")
    parser.add_argument("--client_port", type=int, default=7001)
    parser.add_argument("--use_rb", action="store_true")
    parser.add_argument("--use_mj_fk", action="store_true")
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--use_hand", action="store_true")
    parser.add_argument("--hand_type", type=str, default="inspire_hand", choices=["inspire_hand", "unitree_dex3", "brainco_hand"])
    parser.add_argument("--record_data", action="store_true")
    parser.add_argument("--record_data_path", type=str, default="./recordings/")
    parser.add_argument("--record_data_interval", type=float, default=0.02)
    args = parser.parse_args()
    main(args)
    