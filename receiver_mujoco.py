import socket
import json
import argparse
import time
from collections import deque
from threading import Thread
from typing import Any
from pytorch_kinematics.transforms import quaternion_multiply, quaternion_apply
import torch

class MocapReceiver:
    '''
    This class is used to receive the data from the mocap system.
    '''
    def __init__(self, args=None, server_ip='127.0.0.1', server_port=7001, 
                data_rate=50, num_history=1):
        '''
        input:
            args: arguments
            args.client_ip: the IP address of the client
            args.client_port: the port of the client
        output: None
        '''
        if args is not None and hasattr(args, 'server_ip') and args.server_ip:
            self.UDP_IP = args.server_ip
        else:
            self.UDP_IP = server_ip
        if args is not None and hasattr(args, 'server_port') and args.server_port:
            self.UDP_PORT = args.server_port
        else:
            self.UDP_PORT = server_port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.001)

        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        self.data_rate = data_rate
        self.latest_data = None
        self.num_history = num_history

        self.datas = deque(maxlen=num_history)
        self.delta_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def launch(self):
        self.t = Thread(target=self.receive_data)
        self.t.start()

    def receive_data(self):
        '''
        input: None
        output: None
        '''
        last_time = time.monotonic()
        while True:
            try:
                data = self.sock.recv(1024*64)
            except socket.timeout:
                data = None
            
            if data is not None:
                sensor_data = json.loads(data.decode())
                for key, value in sensor_data.items():
                    if key in ['body_pos', 'body_quat', 'body_lin_vel', 'body_ang_vel']:
                        sensor_data[key] = torch.tensor(value).float()
                self.latest_data = sensor_data

            if self.latest_data is not None and time.monotonic() - last_time > 1.0 / self.data_rate:
                last_time = time.monotonic()
                self.datas.append(self.latest_data)

    def get_data(self):
        '''
        input: None
        output: sensor_data: the retargeted data from the mocap system
        '''
        if len(self.datas) == 0:
            return None

        data_list = list[Any](self.datas)
        if len(self.datas) < self.num_history:
            data_list = [data_list[0]] * (self.num_history - len(self.datas)) + data_list

        # stack data
        data = {}
        for key in data_list[0].keys():
            if isinstance(data_list[0][key], torch.Tensor):
                data[key] = torch.stack([data[key] for data in data_list], dim=0)
            else:
                data[key] = [data[key] for data in data_list]

        # compute the velocity
        if 'body_pos' in data:
            data['body_pos'] = quaternion_apply(self.delta_quat[None, None, :], data['body_pos'])
            body_lin_vel = torch.gradient(data['body_pos'], spacing=1/self.data_rate, dim=0, edge_order=1)[0]
            data['body_lin_vel'] = body_lin_vel

        if 'body_quat' in data:
            data['body_quat'] = quaternion_multiply(self.delta_quat[None, None, :], data['body_quat'])
        return data

    def set_delta_quat(self, quat: torch.Tensor):
        assert quat.shape == self.delta_quat.shape
        self.delta_quat = quat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7001)
    mocap_reciever = MocapReceiver(parser.parse_args()) 

    while 1:
        mocap_reciever.get_data()

    
            