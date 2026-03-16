import numpy as np
import xrobotoolkit_sdk as xrt

from rich import print

import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_pose_to_new_frame(pos_old, quat_old):
    """
    Convert pose from:
        old frame: x-forward, y-up, z-right
    to:
        new frame: x-forward, y-left, z-up

    Parameters
    ----------
    pos_old : (3,) array
        Position [x, y, z] in old coordinate frame
    quat_old : (4,) array
        Quaternion [qw, qx, qy, qz] in old frame (wxyz format)

    Returns
    -------
    pos_new : (3,)
        Position in new frame
    quat_new : (4,)
        Quaternion in new frame (wxyz format: [qw, qx, qy, qz])
    """

    # --- Position transform ---
    pos_new = np.array([
        pos_old[0],
        -pos_old[2],
        pos_old[1]
    ])

    # --- Rotation transform ---
    R_old = R.from_quat(quat_old, scalar_first=True).as_matrix()

    R_coord = np.array([
        [1,  0,  0],   # x_new
        [0,  0, -1],   # y_new
        [0,  1,  0]    # z_new
    ])


    R_new = R_coord @ R_old @ R_coord.T

    R_new = R_new @ R.from_euler('xyz', np.array([0,0,np.pi/2])).as_matrix()

    quat_xyzw = R.from_matrix(R_new).as_quat()   # [qx, qy, qz, qw]

    # --- Convert xyzw → wxyz ---
    quat_new = np.array([
        quat_xyzw[3],   # qw
        quat_xyzw[0],   # qx
        quat_xyzw[1],   # qy
        quat_xyzw[2]    # qz
    ])

    return pos_new, quat_new




class XrClient:
    """Client for the XrClient SDK to interact with XR devices."""

    def __init__(self):
        """Initializes the XrClient and the SDK."""
        xrt.init()
        print("XRoboToolkit SDK initialized.")

    def get_pose_by_name(self, name: str) -> np.ndarray:
        """Returns the pose of the specified device by name.
        Valid names: "left_controller", "right_controller", "headset".
        Pose is [x, y, z, qx, qy, qz, qw]."""
        if name == "left_controller":
            return xrt.get_left_controller_pose()
        elif name == "right_controller":
            return xrt.get_right_controller_pose()
        elif name == "headset":
            return xrt.get_headset_pose()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'left_controller', 'right_controller', 'headset'."
            )

    def get_key_value_by_name(self, name: str) -> float:
        """Returns the trigger/grip value by name (float).
        Valid names: "left_trigger", "right_trigger", "left_grip", "right_grip".
        """
        if name == "left_trigger":
            return xrt.get_left_trigger()
        elif name == "right_trigger":
            return xrt.get_right_trigger()
        elif name == "left_grip":
            return xrt.get_left_grip()
        elif name == "right_grip":
            return xrt.get_right_grip()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'left_trigger', 'right_trigger', 'left_grip', 'right_grip'."
            )

    def get_button_state_by_name(self, name: str) -> bool:
        """Returns the button state by name (bool).
        Valid names: "A", "B", "X", "Y",
                      "left_menu_button", "right_menu_button",
                      "left_axis_click", "right_axis_click"
        """
        if name == "A":
            return xrt.get_A_button()
        elif name == "B":
            return xrt.get_B_button()
        elif name == "X":
            return xrt.get_X_button()
        elif name == "Y":
            return xrt.get_Y_button()
        elif name == "left_menu_button":
            return xrt.get_left_menu_button()
        elif name == "right_menu_button":
            return xrt.get_right_menu_button()
        elif name == "left_axis_click":
            return xrt.get_left_axis_click()
        elif name == "right_axis_click":
            return xrt.get_right_axis_click()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'A', 'B', 'X', 'Y', "
                "'left_menu_button', 'right_menu_button', 'left_axis_click', 'right_axis_click'."
            )

    def get_timestamp_ns(self) -> int:
        """Returns the current timestamp in nanosecon
ds (int)."""
        return xrt.get_time_stamp_ns()

    def get_hand_tracking_state(self, hand: str) -> np.ndarray | None:
        """Returns the hand tracking state for the specified hand.
        Valid hands: "left", "right".
        State is a 27 x 7 numpy array, where each row is [x, y, z, qx, qy, qz, qw] for each joint.
        Returns None if hand tracking is inactive (low quality).
        """
        if hand.lower() == "left":
            if not xrt.get_left_hand_is_active():
                return None
            return xrt.get_left_hand_tracking_state()
        elif hand.lower() == "right":
            if not xrt.get_right_hand_is_active():
                return None
            return xrt.get_right_hand_tracking_state()
        else:
            raise ValueError(f"Invalid hand: {hand}. Valid hands are: 'left', 'right'.")

    def get_joystick_state(self, controller: str) -> list[float]:
        """Returns the joystick state for the specified controller.
        Valid controllers: "left", "right".
        State is a list with shape (2) representing [x, y] for each joystick.
        """
        if controller.lower() == "left":
            return xrt.get_left_axis()
        elif controller.lower() == "right":
            return xrt.get_right_axis()
        else:
            raise ValueError(f"Invalid controller: {controller}. Valid controllers are: 'left', 'right'.")

    def get_motion_tracker_data(self) -> dict:
        """Returns a dictionary of motion tracker data, where the keys are the tracker serial numbers.
        Each value is a dictionary containing the pose, velocity, and acceleration of the tracker.
        """
        num_motion_data = xrt.num_motion_data_available()
        if num_motion_data == 0:
            return {}

        poses = xrt.get_motion_tracker_pose()
        velocities = xrt.get_motion_tracker_velocity()
        accelerations = xrt.get_motion_tracker_acceleration()
        serial_numbers = xrt.get_motion_tracker_serial_numbers()

        tracker_data = {}
        for i in range(num_motion_data):
            serial = serial_numbers[i]
            tracker_data[serial] = {
                "pose": poses[i],
                "velocity": velocities[i],
                "acceleration": accelerations[i],
            }

        return tracker_data

    def get_body_tracking_data(self) -> dict | None:
        """Returns complete body tracking data or None if unavailable.

        Returns:
            Dict with keys: 'poses', 'velocities', 'accelerations', 'imu_timestamps', 'body_timestamp'
            - poses: (24, 7) array [x,y,z,qx,qy,qz,qw] for each joint
            - velocities: (24, 6) array [vx,vy,vz,wx,wy,wz] for each joint
            - accelerations: (24, 6) array [ax,ay,az,wax,way,waz] for each joint

            joint_names = [
                "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
                "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
                "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
                "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand"
            ]

        """
        if not xrt.is_body_data_available():
            return None

        return {
            "poses": xrt.get_body_joints_pose(),
            "velocities": xrt.get_body_joints_velocity(),
            "accelerations": xrt.get_body_joints_acceleration(),
        }

    def get_body_tracking_human_frame(self, data: dict | None = None) -> dict | None:
        if data is None:
            data = self.get_body_tracking_data()
        if data is None or "poses" not in data:
            return None
        poses = data["poses"]
        xr_joint_names = [
            "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
            "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
            "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
            "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand"
        ]
        name_to_idx = {n: i for i, n in enumerate(xr_joint_names)}
        mapping = {
            "Pelvis": "pelvis",
            "Spine3": "spine3",
            "Spine1": "spine1",
            "Spine2": "spine2",
            "Left_Hip": "left_hip",
            "Right_Hip": "right_hip",
            "Left_Knee": "left_knee",
            "Right_Knee": "right_knee",
            "Left_Foot": "left_foot",
            "Right_Foot": "right_foot",
            "Left_Shoulder": "left_shoulder",
            "Right_Shoulder": "right_shoulder",
            "Left_Elbow": "left_elbow",
            "Right_Elbow": "right_elbow",
            "Left_Wrist": "left_wrist",
            "Right_Wrist": "right_wrist",
            "Head" : "head"
        }
        out = {}
        for xr_name, gmr_name in mapping.items():
            idx = name_to_idx.get(xr_name)
            if idx is None:
                continue
            arr = poses[idx]
            pos = np.asarray(arr[:3], dtype=float)
            quat = np.asarray([arr[6], arr[3], arr[4], arr[5]], dtype=float) # xyzw -> wxyz
            out[gmr_name] = [pos, quat]

        for gmr_name, arr in out.items():
            pos_old = arr[0]
            quat_old = arr[1]
            pos_new, quat_new = convert_pose_to_new_frame(pos_old, quat_old) # wxyz
            pos_new += np.array([0.0, 0.0, 1.6])
            out[gmr_name] = [pos_new, quat_new]
        return out

    def close(self):
        xrt.close()

if __name__ == "__main__":
    import time

    client = XrClient()
    while True:
        pose = client.get_body_tracking_data()

        if pose is not None:
            print("Body tracking data:", pose)
        else:
            print("No body tracking data available.")

        time.sleep(0.1)
