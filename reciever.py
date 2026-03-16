import socket
import json
import argparse


class Mocap_reciever:
    '''
    This class is used to receive the data from the mocap system.
    '''
    def __init__(self, server_ip='127.0.0.1', server_port=7001):
        '''
        input:
            args: arguments
            args.client_ip: the IP address of the client
            args.client_port: the port of the client
        output: None
        '''
        
        self.UDP_IP = server_ip
        self.UDP_PORT = server_port
            
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))

    def get_data(self):
        '''
        input: None
        output: sensor_data: the retargeted data from the mocap system
        '''
        data, addr = self.sock.recvfrom(1024*16)
        sensor_data = json.loads(data.decode())
        return sensor_data


if __name__ == "__main__":
    mocap_reciever = Mocap_reciever() 

    while True:
        mocap_reciever.get_data()

    
            