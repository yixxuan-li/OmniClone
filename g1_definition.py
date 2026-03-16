JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

JOINT_LIMITS = [
    (-2.5307, 2.8798),         # left_hip_pitch_joint
    (-0.5236, 2.9671),         # left_hip_roll_joint
    (-2.7576, 2.7576),         # left_hip_yaw_joint
    (-0.087267, 2.8798),       # left_knee_joint
    (-10.87267, 10.5236),        # left_ankle_pitch_joint
    (-10.2618, 10.2618),         # left_ankle_roll_joint
    (-2.5307, 2.8798),         # right_hip_pitch_joint
    (-2.9671, 0.5236),         # right_hip_roll_joint
    (-2.7576, 2.7576),         # right_hip_yaw_joint
    (-0.087267, 2.8798),       # right_knee_joint
    (-10.87267, 10.5236),        # right_ankle_pitch_joint
    (-10.2618, 10.2618),         # right_ankle_roll_joint
    (-2.618, 2.618),           # waist_yaw_joint
    (-0.52, 0.52),             # waist_roll_joint
    (-0.52, 0.52),             # waist_pitch_joint
    (-3.0892, 2.6704),         # left_shoulder_pitch_joint
    (-1.5882, 2.2515),         # left_shoulder_roll_joint
    (-2.618, 2.618),           # left_shoulder_yaw_joint
    (-1.0472, 2.0944),         # left_elbow_joint
    (-1.972222054, 1.972222054), # left_wrist_roll_joint
    (-1.614429558, 1.614429558), # left_wrist_pitch_joint
    (-1.614429558, 1.614429558), # left_wrist_yaw_joint
    (-3.0892, 2.6704),         # right_shoulder_pitch_joint
    (-2.2515, 1.5882),         # right_shoulder_roll_joint
    (-2.618, 2.618),           # right_shoulder_yaw_joint
    (-1.0472, 2.0944),         # right_elbow_joint
    (-1.972222054, 1.972222054), # right_wrist_roll_joint
    (-1.614429558, 1.614429558), # right_wrist_pitch_joint
    (-1.614429558, 1.614429558)  # right_wrist_yaw_joint
]

TORQUE_LIMITS = [
    88,                   # left_hip_pitch_joint
    139,                  # left_hip_roll_joint
    88,                   # left_hip_yaw_joint
    139,                  # left_knee_joint
    50,                   # left_ankle_pitch_joint
    50,                   # left_ankle_roll_joint
    88,                   # right_hip_pitch_joint
    139,                  # right_hip_roll_joint
    88,                   # right_hip_yaw_joint
    139,                  # right_knee_joint
    50,                   # right_ankle_pitch_joint
    50,                   # right_ankle_roll_joint
    88,                   # waist_yaw_joint
    50,                   # waist_roll_joint
    50,                   # waist_pitch_joint
    25,                   # left_shoulder_pitch_joint
    25,                   # left_shoulder_roll_joint
    25,                   # left_shoulder_yaw_joint
    25,                   # left_elbow_joint
    25,                   # left_wrist_roll_joint
    5,                    # left_wrist_pitch_joint
    5,                    # left_wrist_yaw_joint
    25,                   # right_shoulder_pitch_joint
    25,                   # right_shoulder_roll_joint
    25,                   # right_shoulder_yaw_joint
    25,                   # right_elbow_joint
    25,                   # right_wrist_roll_joint
    5,                    # right_wrist_pitch_joint
    5                     # right_wrist_yaw_joint
]

EEF_LINKS = [
    'pelvis',
    'torso_link',
    'left_shoulder_pitch_link',
    'left_shoulder_roll_link',
    'left_shoulder_yaw_link',
    'left_elbow_link',
    'left_wrist_roll_link',
    'left_wrist_pitch_link',
    'left_wrist_yaw_link',
    'right_shoulder_pitch_link',
    'right_shoulder_roll_link',
    'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link',
    'right_wrist_pitch_link',
    'right_wrist_yaw_link',
    'left_hip_pitch_link',
    'left_hip_roll_link',
    'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_roll_link',
    'right_hip_pitch_link',
    'right_hip_roll_link',
    'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_roll_link',
]

import torch
def resolve_kpkd(hip_kp, hip_kd, knee_kp, knee_kd, ankle_kp, ankle_kd,
                 waist_kp, waist_kd, shoulder_kp, shoulder_kd,
                 arm_kp, arm_kd, wrist_kp, wrist_kd, rest_kps=None, rest_kds=None):
    kp = torch.zeros(len(JOINT_NAMES))
    kd = torch.zeros(len(JOINT_NAMES))

    for i, name in enumerate(JOINT_NAMES):
        if 'hip' in name:
            kp[i] = hip_kp
            kd[i] = hip_kd
        elif 'knee' in name:
            kp[i] = knee_kp
            kd[i] = knee_kd
        elif 'ankle' in name:
            kp[i] = ankle_kp
            kd[i] = ankle_kd
        elif 'waist' in name:
            kp[i] = waist_kp
            kd[i] = waist_kd
        elif 'shoulder_pitch' in name or 'shoulder_roll' in name:
            kp[i] = shoulder_kp
            kd[i] = shoulder_kd
        elif 'shoulder_yaw' in name or 'elbow' in name or 'wrist_roll' in name:
            kp[i] = arm_kp
            kd[i] = arm_kd
        elif 'wrist_pitch' in name or 'wrist_yaw' in name:
            kp[i] = wrist_kp
            kd[i] = wrist_kd
    
    if rest_kps is not None:
        for k, v in rest_kps.items():
            if k in JOINT_NAMES:
                i = JOINT_NAMES.index(k)
                kp[i] = v
    if rest_kds is not None:
        for k, v in rest_kds.items():
            if k in JOINT_NAMES:
                i = JOINT_NAMES.index(k)
                kd[i] = v
    return kp, kd
