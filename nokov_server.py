__author__ = 'duguguang'

from nokov.nokovsdk import *
from nokov_python_types import *
from ctypes import pointer
from threading import Thread
import numpy as np
from time import sleep 
from typing import List

from scipy.spatial.transform import Rotation as R


preFrmNo = 0
curFrmNo = 0

# set the offset here based on your own data
offset = 0

RIGID_BODY_ID_MAP = {
    1+offset: "Hips",
    4+offset: "Spine",
    5+offset: "Spine1",
    14+offset: "Neck",
    15+offset: "Head",
    6+offset: "LeftShoulder",
    7+offset: "LeftArm",
    8+offset: "LeftForeArm",
    9+offset: "LeftHand",
    10+offset: "RightShoulder",
    11+offset: "RightArm",
    12+offset: "RightForeArm",
    13+offset: "RightHand",
    16+offset: "LeftUpLeg",
    17+offset: "LeftLeg",
    18+offset: "LeftFoot",
    19+offset: "LeftToeBase",
    20+offset: "RightUpLeg",
    21+offset: "RightLeg",
    22+offset: "RightFoot",
    23+offset: "RightToeBase",
    # 22+offset: "LeftHandThumb1",
    # 23+offset: "LeftHandThumb2",
    # 24+offset: "LeftHandThumb3",
    # 25+offset: "LeftHandIndex1",
    # 26+offset: "LeftHandIndex2",
    # 27+offset: "LeftHandIndex3",
    # 28+offset: "LeftHandMiddle1",
    # 29+offset: "LeftHandMiddle2",
    # 30+offset: "LeftHandMiddle3",
    # 31+offset: "LeftHandRing1",
    # 32+offset: "LeftHandRing2",
    # 33+offset: "LeftHandRing3",
    # 34+offset: "LeftHandPinky1",
    # 35+offset: "LeftHandPinky2",
    # 36+offset: "LeftHandPinky3",
    # 37+offset: "RightHandThumb1",
    # 38+offset: "RightHandThumb2",
    # 39+offset: "RightHandThumb3",
    # 40+offset: "RightHandIndex1",
    # 41+offset: "RightHandIndex2",
    # 42+offset: "RightHandIndex3",
    # 43+offset: "RightHandMiddle1",
    # 44+offset: "RightHandMiddle2",
    # 45+offset: "RightHandMiddle3",
    # 46+offset: "RightHandRing1",
    # 47+offset: "RightHandRing2",
    # 48+offset: "RightHandRing3",
    # 49+offset: "RightHandPinky1",
    # 50+offset: "RightHandPinky2",
    # 51+offset: "RightHandPinky3"
}

# Create hand id map (without "Right" prefix)
HAND_ID_MAP = {
    0: "Palm",          # Palm
    1: "Thumb1",
    2: "Thumb2",
    3: "Thumb3",
    4: "Index1",
    5: "Index2",
    6: "Index3",
    7: "Middle1",
    8: "Middle2",
    9: "Middle3",
    10: "Ring1",
    11: "Ring2",
    12: "Ring3",
    13: "Pinky1",
    14: "Pinky2",
    15: "Pinky3"
}


class Nokov_Server:
    
    def __init__(self, server_ip = '192.168.123.100'):
        self.serverIp = server_ip


        print ('serverIp is %s' % self.serverIp)
        print("Started the Nokov_SDK_Client")
        client = PySDKClient()

        ver = client.PyNokovVersion()
        print('NokovSDK Sample Client 4.0.0.5645(NokovSDK ver. %d.%d.%d.%d)' % (ver[0], ver[1], ver[2], ver[3]))
        print("Begin to init the SDK Client")
        ret = client.Initialize(bytes(self.serverIp, encoding = "utf8"))
        if ret == 0:
            print("Connect to the Nokov Succeed")
        else:
            print("Connect Failed: [%d]" % ret)
            exit(0)

        
        # if ch == 2:
        self.latest_frame = None
        self.latest_left_hand = None
        self.latest_right_hand = None
        self.latest_rigid_bodies = dict[str, np.ndarray]()
        t = Thread(target=self.read_data_func, args=(client,))
        t.start()
        self.client = client
        sleep(1)

    def get_latest_rigid_body(self):
        return self.latest_rigid_bodies
        
    def get_latest_hand(self):
        data = np.zeros((2, 6, 3))
        if self.latest_left_hand is not None:
            data[0, :, :] = self.latest_left_hand
        if self.latest_right_hand is not None:
            data[1, :, :] = self.latest_right_hand
        return data
    
    def get_latest_frame(self):
    #     # return self.parse_skeleton(self.frame)
        frame = {}
        if self.latest_frame is not None:
            for id, key in enumerate(self.latest_frame.keys()):
                if key in RIGID_BODY_ID_MAP.keys():
                    frame[RIGID_BODY_ID_MAP[key]] = [self.latest_frame[key]['pos'], self.latest_frame[key]['rot']]
        
            return frame
    #         frameData = self.frame.contents
    
    def parse_skeleton(self, pFrameOfMocapData):
        if pFrameOfMocapData == None:
            print("Not get the data frame.\n")
        else:
            frameData = pFrameOfMocapData.contents
            length = 128
            
            print(f"Markerset.Skeletons [Count={frameData.nSkeletons}]")
            for iSkeleton in range(frameData.nSkeletons):
                # Segments
                skeleton = frameData.Skeletons[iSkeleton]
                print(f"nSegments Count={skeleton.nRigidBodies}")
                print("{")
                for iBody in range(skeleton.nRigidBodies):
                    body = skeleton.RigidBodyData[iBody]
                    print(f"\tSegment id:{body.ID}")
                    print(f"\t    (mm)\tx:{body.x:6.2f}\ty:{body.y:6.2f}\tz:{body.z:6.2f}")
                    print(f"\t\t\tqx:{body.qx:6.2f}\tqy:{body.qy:6.2f}\tqz:{body.qz:6.2f}\tqz:{body.qw:6.2f}")
                    for iMarkerIndex in range(body.nMarkers):
                        marker = body.Markers[iMarkerIndex]
                        print(f"\tMarker{body.MarkerIDs[iMarkerIndex]}(mm)"\
                            f"\tx:{marker[0]:6.2f}\ty:{marker[1]:6.2f}\tz:{marker[2]:6.2f}")
                print("}\n")
            
            
        
        
    def client_run(self):
        while True:
            client.PyGetDataDescriptions(self.pdds)


    def py_data_func(self, pFrameOfMocapData, pUserData):
        if pFrameOfMocapData == None:  
            print("Not get the data frame.\n")
        else:
            frameData = pFrameOfMocapData.contents
            global preFrmNo, curFrmNo 
            curFrmNo = frameData.iFrame
            if curFrmNo == preFrmNo:
                return
            preFrmNo = curFrmNo

            length = 128
            szTimeCode = bytes(length)
            frameExtData = frameData.FrameExtendData
            rigidBodyExtData = None
            for i in range(frameExtData.nExtendDataNum):
                if frameExtData.extendData[i].type == ExtendDataType.ExtendDataRigidBody.value:
                    rigidBodyExtData = frameExtData.extendData[i]
                    break

            self.client.PyTimecodeStringify(frameData.Timecode, frameData.TimecodeSubframe, szTimeCode, length)
            # print(f"\nFrameNo: {frameData.iFrame}\tTimeStamp:{frameData.iTimeStamp}\t Timecode:{szTimeCode.decode('utf-8')}")					
            # print( f"MarkerSet [Count={frameData.nMarkerSets}]")

            # print(f"Markerset.Skeletons [Count={frameData.nSkeletons}]")
            for iSkeleton in range(frameData.nSkeletons):
                # Segments
                skeleton = frameData.Skeletons[iSkeleton]
                if skeleton.nRigidBodies == 16:
                    self.process_hand_data(skeleton)
                if skeleton.nRigidBodies == 23 or skeleton.nRigidBodies == 53:
                    self.process_human_data(skeleton)
                    if skeleton.nRigidBodies == 53:
                        pass

            for iRigidBody in range(frameData.nRigidBodies):
                rigid_body = frameData.RigidBodies[iRigidBody]
                self.process_rigid_body_data(rigid_body)

    
    def process_rigid_body_data(self, rigid_body):
        # Current frame: y-up, x-forward, z-right
        # Target frame: z-up, x-forward, y-left
        # Position conversion: [x_old, y_old, z_old] -> [x_old, -z_old, y_old]
        pos_old = np.array([rigid_body.x / 1000.0, rigid_body.y / 1000.0, rigid_body.z / 1000.0])
        pos_new = np.array([pos_old[0], -pos_old[2], pos_old[1]])
        
        # Quaternion conversion for coordinate frame change
        # Old: [qw, qx, qy, qz] where x-forward, y-up, z-right
        # New: [qw, qx, qy, qz] where x-forward, y-left, z-up
        quat_old = np.array([rigid_body.qx, rigid_body.qy, rigid_body.qz, rigid_body.qw])
        
        # Convert to rotation matrix
        R_old = R.from_quat(quat_old).as_matrix()
        
        # Create transformation matrix to convert from old frame to new frame
        # Old: x-forward, y-up, z-right
        # New: x-forward, y-left, z-up
        # 
        # Express new axes in old frame coordinates:
        # x_new = [1, 0, 0] (forward unchanged)
        # y_new = [0, 0, -1] (left = -right)
        # z_new = [0, 1, 0] (up)
        R_coord = np.array([
            [1,  0,  0 ],  # x_new in old coordinates
            [0,  0, -1 ],  # y_new in old coordinates
            [0,  1,  0 ]   # z_new in old coordinates
        ])
        
        # Transform rotation from old frame to new frame
        R_new = R_coord @ R_old @ R_coord.T

        quat_new = R.from_matrix(R_new).as_quat()
        
        rb_pose = np.array([pos_new[0], pos_new[1], pos_new[2], 
                                quat_new[3], quat_new[0], quat_new[1], quat_new[2]])
        
        rb_id = rigid_body.ID
        rb_name = self.rigid_body_desc[rb_id].szName
        self.latest_rigid_bodies[rb_name] = rb_pose

    def process_hand_data(self, skeleton):
        desc: PySkeletonDescription = self.skeleton_desc[skeleton.skeletonID]
        hand_type = desc.RigidBodies[0].szName.lower().replace('hand', '')

        rb_datas = []
        ids = [0, 3, 6, 9, 12, 15]
        for id in ids:
            rb_datas.append(skeleton.RigidBodyData[id])
        rb_pos = np.array([[rb.x, rb.y, rb.z] for rb in rb_datas]) / 1000.0
        rb_quat = np.array([-rb_datas[0].qx, -rb_datas[0].qy, -rb_datas[0].qz, rb_datas[0].qw])
        rb_pos = R.from_quat(rb_quat).apply(rb_pos - rb_pos[0:1])

        if hand_type == 'left':
            # convert to unitree convention for left hand
            # Current: x(wrist->index), y(front->back), z(pinky->index)
            # Unitree: x(palm->back), y(middle->wrist), z(pinky->index)
            rb_pos_converted = np.zeros_like(rb_pos)
            rb_pos_converted[:, 0] = rb_pos[:, 1]  # y -> x (front->back)
            rb_pos_converted[:, 1] = -rb_pos[:, 0]  # -x -> y (wrist->index becomes middle->wrist)
            rb_pos_converted[:, 2] = rb_pos[:, 2]  # z -> z (pinky->index, same direction)
            rb_pos = rb_pos_converted
            self.latest_left_hand = rb_pos

        elif hand_type == 'right':
            # convert to unitree convention
            # Current: x(middle->wrist), y(front->back), z(pinky->index)
            # Unitree: x(palm->back), y(middle->wrist), z(index->pinky)
            rb_pos_converted = np.zeros_like(rb_pos)
            rb_pos_converted[:, 0] = rb_pos[:, 1]  # y -> x (front->back)
            rb_pos_converted[:, 1] = rb_pos[:, 0]  # x -> y (middle->wrist)
            rb_pos_converted[:, 2] = -rb_pos[:, 2]  # -z -> z (pinky->index becomes index->pinky)
            rb_pos = rb_pos_converted
            self.latest_right_hand = rb_pos
        
        else:
            raise ValueError(f"Invalid hand type: {hand_type}")
        
    def process_hand_intergated_data(self, skeleton):
        desc: PySkeletonDescription = self.skeleton_desc[skeleton.skeletonID]
        hand_type = desc.RigidBodies[0].szName.lower().replace('hand', '')

        rb_datas = []
        ids = [0, 3, 6, 9, 12, 15]
        for id in ids:
            rb_datas.append(skeleton.RigidBodyData[id])
        rb_pos = np.array([[rb.x, rb.y, rb.z] for rb in rb_datas]) / 1000.0
        rb_quat = np.array([-rb_datas[0].qx, -rb_datas[0].qy, -rb_datas[0].qz, rb_datas[0].qw])
        rb_pos = R.from_quat(rb_quat).apply(rb_pos - rb_pos[0:1])

        if hand_type == 'left':
            # convert to unitree convention for left hand
            # Current: x(wrist->index), y(front->back), z(pinky->index)
            # Unitree: x(palm->back), y(middle->wrist), z(pinky->index)
            rb_pos_converted = np.zeros_like(rb_pos)
            rb_pos_converted[:, 0] = rb_pos[:, 1]  # y -> x (front->back)
            rb_pos_converted[:, 1] = -rb_pos[:, 0]  # -x -> y (wrist->index becomes middle->wrist)
            rb_pos_converted[:, 2] = rb_pos[:, 2]  # z -> z (pinky->index, same direction)
            rb_pos = rb_pos_converted
            self.latest_left_hand = rb_pos

        elif hand_type == 'right':
            # convert to unitree convention
            # Current: x(middle->wrist), y(front->back), z(pinky->index)
            # Unitree: x(palm->back), y(middle->wrist), z(index->pinky)
            rb_pos_converted = np.zeros_like(rb_pos)
            rb_pos_converted[:, 0] = rb_pos[:, 1]  # y -> x (front->back)
            rb_pos_converted[:, 1] = rb_pos[:, 0]  # x -> y (middle->wrist)
            rb_pos_converted[:, 2] = -rb_pos[:, 2]  # -z -> z (pinky->index becomes index->pinky)
            rb_pos = rb_pos_converted
            self.latest_right_hand = rb_pos
        
        else:
            raise ValueError(f"Invalid hand type: {hand_type}")
                

    def process_human_data(self, skeleton):
        latest_frame = {}
        # print(f"nSegments Count={skeleton.nRigidBodies}")
        # print("{")
        for iBody in range(23):
            body = skeleton.RigidBodyData[iBody]
            body_state = {}
            body_state['pos'] = np.array([body.x / 1000, -body.z / 1000, body.y / 1000])
            # body_state['pos'] = [body.x / 1000, body.y / 1000, body.z / 1000]
            
            body_quat = np.array([body.qx, body.qy, body.qz, body.qw])

            body_rotation_matrix = R.from_quat(body_quat).as_matrix()
            R_align = R.from_euler('xyz', np.array([np.pi/2,0,0])).as_matrix()
            body_rotation_matrix =  R_align @ body_rotation_matrix @ R_align.T
            body_rotation_matrix =  body_rotation_matrix @ R.from_euler('xyz', np.array([0,0,-np.pi/2])).as_matrix()
            # body_rotation_matrix = body_rotation_matrix @ R_align
            body_quat = R.from_matrix(body_rotation_matrix).as_quat()
            
            body_state['rot'] = np.roll(body_quat, 1)
            latest_frame[body.ID] = body_state

        self.latest_frame = latest_frame



    def py_msg_func(self, iLogLevel, szLogMessage):
        szLevel = "None"
        if iLogLevel == 4:
            szLevel = "Debug"
        elif iLogLevel == 3:
            szLevel = "Info"
        elif iLogLevel == 2:
            szLevel = "Warning"
        elif iLogLevel == 1:
            szLevel = "Error"
    
        print("[%s] %s" % (szLevel, cast(szLogMessage, c_char_p).value))


    def py_analog_channel_func(self, pAnalogData, pUserData):
        if pAnalogData == None:  
            print("Not get the analog data frame.\n")
            pass
        else:
            anData = pAnalogData.contents
            print(f"\nFrameNO:{anData.iFrame}\tTimeStamp:{anData.iTimeStamp}")
            print(f"Analog Channel Number:{anData.nAnalogdatas}, SubFrame: {anData.nSubFrame}")
            for ch in range(anData.nAnalogdatas):
                print(f"Channel {ch} ", end="")
                for sub in range(anData.nSubFrame):
                    print(f",{anData.Analogdata[ch][sub]:6.4f}", end="")
                print("")

    def read_data_func(self, client):
        self.skeleton_desc = {}
        self.rigid_body_desc = {}
        while True:
            data_descripton = pointer(DataDescriptions())
            self.frame = client.PyGetLastFrameOfMocapData()
            client.PyGetDataDescriptions(data_descripton)

            self.n_data_desc = data_descripton.contents.nDataDescriptions
            self.data_desc = data_descripton.contents.arrDataDescriptions
            for i in range(self.n_data_desc):
                desc = self.data_desc[i]
                if desc.type == 1:
                    desc = desc.Data.RigidBodyDescription.contents
                    self.rigid_body_desc[desc.ID] = PyRigidBodyDescription(desc.szName, desc.ID, desc.parentID, desc.offsetx, desc.offsety, desc.offsetz, desc.qx, desc.qy, desc.qz, desc.qw)
                elif desc.type == 2:
                    desc = desc.Data.SkeletonDescription.contents
                    RigidBodies = []
                    for j in range(desc.nRigidBodies):
                        rb = desc.RigidBodies[j]
                        RigidBodies.append(PyRigidBodyDescription(rb.szName, rb.ID, rb.parentID, rb.offsetx, rb.offsety, rb.offsetz, rb.qx, rb.qy, rb.qz, rb.qw))
                    skeleton_desc = PySkeletonDescription(desc.szName, desc.skeletonID, desc.nRigidBodies, RigidBodies)
                    self.skeleton_desc[desc.skeletonID] = skeleton_desc
            
            if self.frame:
                try:
                    self.py_data_func(self.frame, client)
                finally:
                    client.PyNokovFreeFrame(self.frame)

    def py_desc_func(self, pdds):
        dataDefs = pdds.contents

        for iDef in range(dataDefs.nDataDescriptions):
            dataDef = dataDefs.arrDataDescriptions[iDef]
            
            if dataDef.type == DataDescriptors.Descriptor_Skeleton.value:
                skeletonDef = dataDef.Data.SkeletonDescription.contents
                print(f"Skeleton Name:{skeletonDef.szName.decode('utf-8')}, id:{skeletonDef.skeletonID}, rigids:{skeletonDef.nRigidBodies}")
                for iBody in range(skeletonDef.nRigidBodies):
                    bodyDef = skeletonDef.RigidBodies[iBody]
                    print(f"[{bodyDef.ID}] {bodyDef.szName.decode('utf-8')} {bodyDef.parentID} "\
                        f"{bodyDef.offsetx:.6f}mm {bodyDef.offsety:.6f}mm {bodyDef.offsetz:.6f}mm "\
                        f"{bodyDef.qx:.6f} {bodyDef.qy:.6f} {bodyDef.qz:.6f} {bodyDef.qw:.6f}")
            elif dataDef.type == DataDescriptors.Descriptor_MarkerSet.value:
                markerSetDef = dataDef.Data.MarkerSetDescription.contents
                print(f"MarkerSetName: {markerSetDef.szName.decode('utf-8')}")
                for markerIndex in range(markerSetDef.nMarkers):
                    markerName = markerSetDef.szMarkerNames[markerIndex]
                    print(f"Marker[{markerIndex}] : {markerName.decode('utf-8')}")
            elif dataDef.type == DataDescriptors.Descriptor_RigidBody.value:
                rigidBody = dataDef.Data.RigidBodyDescription.contents
                print(f"RigidBody:{rigidBody.szName.decode('utf-8')} ID:{rigidBody.ID}")
            elif dataDef.type == DataDescriptors.Descriptor_ForcePlate.value:
                forcePlateDef = dataDef.Data.ForcePlateDescription.contents
                for chIndex in range(forcePlateDef.nChannels):
                    channelName = forcePlateDef.szChannelNames[chIndex].value.decode('utf-8')
                    print(f"Channel:{chIndex} {channelName}")
            elif dataDef.type == DataDescriptors.Descriptor_Param.value:
                dataParam = dataDef.Data.DataParam.contents
                print(f'FrameRate:{dataParam.nFrameRate}')

    def py_notify_func(self, pNotify, userData):
        notify = pNotify.contents
        print(f"\nNotify Type: {notify.nType}, Value: {notify.nValue}, "\
            f"timestamp:{notify.nTimeStamp}, msg: '{notify.sMsg.decode('utf-8')}', "\
            f"param1:{notify.nParam1}, param2:{notify.nParam2}, param3:{notify.nParam3}, param4:{notify.nParam4}")


 
if __name__ == "__main__":
    mocap_client = Nokov_Server()
    while True:
        pass
