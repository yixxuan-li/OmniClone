from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import threading
import argparse
import time

from general_motion_retargeting.pico_client import XrClient

def main(args):

    # import time
    # cnt = 0
    # while cnt < 5:
    #     time.sleep(1)
    #     cnt += 1
    #     print(cnt)
    
    client = XrClient()

    retarget = GMR(
        src_human="pico",
        tgt_robot=args.robot,
        actual_human_height=1.7,
    )
    viewer = RobotMotionViewer(robot_type=args.robot, transparent_robot=0,)
    
    save = False
    while True:
        raw_frame = client.get_body_tracking_data()
        frame = client.get_body_tracking_human_frame(raw_frame)

        # debug use: save the frame to a file
        # if not save:
        #     save = True 
        #     import pickle
        #     with open("pico_frame.pkl", "wb") as f:
        #         pickle.dump(raw_frame, f)

        # data_path = "/home/whx/project/pico_teleop/GMR/pico_frame.pkl"
        # import pickle
        # with open(data_path, "rb") as f:
        #     raw_frame = pickle.load(f)
        # frame = client.get_body_tracking_human_frame(raw_frame)

        if frame is None:
            print("No body tracking data available.")
            time.sleep(0.1)
            continue

        qpos = retarget.retarget(frame)
        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            rate_limit=False,
            show_human_body_name=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    args = parser.parse_args()
    main(args)
    
