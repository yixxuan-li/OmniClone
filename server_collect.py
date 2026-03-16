import numpy as np
import multiprocessing
from multiprocessing import Process, Array
from threading import Thread
import torch
import time
from deploy_utils.sim2sim import MujocoEnv
from deploy_utils.sim2real import UnitreeEnv
from deploy_utils.sim2real_head import UnitreeHeadEnv
from deploy_utils.utils import AutoQueue
from g1_definition import (
    resolve_kpkd,
    JOINT_NAMES,
    EEF_LINKS,
    JOINT_LIMITS,
    TORQUE_LIMITS,
)
import math_utils
from receiver_mujoco import MocapReceiver
from vla_data_collection.vla_data_collector import VlaDataCollector
# from deploy_utils.inspire_service import InspireController
from scipy.spatial.transform import Rotation as R
import math
from typing import Union
import pickle
import plotext as plt
from pytorch_kinematics.transforms import quaternion_multiply, quaternion_apply


def quaternion_conjugate(q):
    """
    q: (..., 4) quaternion in (w, x, y, z) format
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)

@torch.inference_mode()
def compute_observation(joint_states, gravities, 
                        root_ang_vels, last_actions,
                        root_quat,
                        motion_rb_pos: torch.Tensor,
                        motion_rb_quat: torch.Tensor,
                        root_lin_vel: torch.Tensor):
    """
    Compute the observation for the policy.
    """
    observations = torch.cat([joint_states.flatten(), gravities.flatten(), root_ang_vels.flatten(), last_actions.flatten()]).clone()

    root_quat_inv = math_utils.quat_conjugate(math_utils.yaw_quat(root_quat[None, None, :]))
    root_quat_inv = root_quat_inv.repeat(motion_rb_pos.shape[0], motion_rb_pos.shape[1], 1)

    delta_global = (motion_rb_pos[:, 0:1] - motion_rb_pos[0:1, 0:1])
    delta_global = math_utils.quat_apply(root_quat_inv[:, 0:1], delta_global)

    dist = torch.norm(delta_global, dim=-1, keepdim=True)
    exceed_scale = (1.0 / (dist + 1e-6)).clamp(max=1.0)
    delta_global = delta_global * exceed_scale

    delta_local = math_utils.quat_apply(root_quat_inv, motion_rb_pos - motion_rb_pos[0:1, 0:1])
    delta_pos = torch.cat([delta_global, delta_local], dim=1).flatten()
    delta_quat = math_utils.quat_mul(root_quat_inv, motion_rb_quat)
    delta_quat = math_utils.matrix_from_quat(delta_quat)[..., :2].flatten() 

    rel_lin_vel = math_utils.quat_apply(root_quat_inv[:, 0], root_lin_vel).flatten()
    task_obs = torch.cat([delta_pos, delta_quat, rel_lin_vel], dim=-1)
    observations = torch.cat([observations, task_obs.reshape(-1)], dim=0)
    return dict(x=observations)

# @torch.inference_mode()
# def compute_observation(joint_states, gravities, 
#                         root_ang_vels, last_actions,
#                         root_quat,
#                         motion_rb_pos: torch.Tensor,
#                         motion_rb_quat: torch.Tensor,
#                         root_lin_vel: torch.Tensor):
#     """
#     Compute the observation for the policy.
#     """
#     observations = torch.cat([joint_states.flatten(), gravities.flatten(), root_ang_vels.flatten(), last_actions.flatten()]).clone()

#     root_quat_inv = math_utils.quat_conjugate(math_utils.yaw_quat(root_quat[None, None, :]))
#     root_quat_inv = root_quat_inv.repeat(motion_rb_pos.shape[0], motion_rb_pos.shape[1], 1)

#     delta_global = (motion_rb_pos[:, [0]] - motion_rb_pos[0:1, 0:1])
#     delta_global = math_utils.quat_apply(root_quat_inv[:, 0:1], delta_global)

#     dist = torch.norm(delta_global, dim=-1, keepdim=True)
#     exceed_scale = (1.0 / (dist + 1e-6)).clamp(max=1.0)
#     delta_global = delta_global * exceed_scale

#     delta_local = math_utils.quat_apply(root_quat_inv, motion_rb_pos - motion_rb_pos[0:1, 0:1])
#     delta_local = torch.cat([delta_local[:, [0]], delta_local[:, 2:2+14]], dim = -2)
#     delta_pos = torch.cat([delta_global, delta_local], dim=1).flatten()

#     rel_lin_vel = math_utils.quat_apply(root_quat_inv[:, 0], root_lin_vel).flatten()
#     task_obs = torch.cat([delta_pos, rel_lin_vel], dim=-1)
#     observations = torch.cat([observations, task_obs.reshape(-1)], dim=0)
#     return dict(x=observations)
def get_relative_root_quat(root_quat, init_root_quat):
    return quaternion_multiply(quaternion_conjugate(init_root_quat), root_quat)

def get_relative_head_angles(head_rot, hip_rot):
    """
    in degree format
    input:
        head_rot: [w, x, y, z] head quat
        hip_rot:  [w, x, y, z] hip quat
    output:
        relative_pitch, relative_yaw
    """

    def to_np(q):
        q = np.asarray(q, dtype=float)
        if q.shape != (4,):
            raise ValueError("Quaternion must be length 4: [w, x, y, z]")
        return q

    def normalize(q):
        n = np.linalg.norm(q)
        if n == 0:
            raise ValueError("Zero-length quaternion provided")
        return q / n

    def quat_conjugate(q):
        w, x, y, z = q
        return np.array([ w, -x, -y, -z ], dtype=float)

    def quat_mul(a, b):
        """Quaternion multiplication a * b, quaternions in [w,x,y,z]."""
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        w = aw*bw - ax*bx - ay*by - az*bz
        x = aw*bx + ax*bw + ay*bz - az*by
        y = aw*by - ax*bz + ay*bw + az*bx
        z = aw*bz + ax*by - ay*bx + az*bw
        return np.array([w, x, y, z], dtype=float)

    # print(f'head_rot: {head_rot}')
    # convert & normalize
    q_head = normalize(to_np(head_rot))
    q_hip  = normalize(to_np(hip_rot))

    # relative quaternion: q_rel = q_hip^{-1} * q_head
    # for unit quaternions inverse == conjugate
    q_rel = quat_mul(quat_conjugate(q_hip), q_head)
    w, x, y, z = q_rel

    # compute Euler angles (roll, pitch, yaw) using standard formulas
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # clamp for numerical stability
    if sinp >= 1.0:
        pitch = math.pi / 2.0
    elif sinp <= -1.0:
        pitch = -math.pi / 2.0
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # convert to degrees
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)

    return pitch_deg, yaw_deg

class Intermediates:
    def __init__(self, num_history, joint_pos, joint_vel, gravity_vec, root_ang_vel, last_action):
        self.gravity_vec = gravity_vec
        self.joint_states = AutoQueue(torch.zeros_like(torch.cat([joint_pos, joint_vel * 0.05], dim=0)), max_size=num_history)
        self.gravities = AutoQueue(torch.zeros_like(gravity_vec), max_size=num_history)
        self.root_ang_vels = AutoQueue(torch.zeros_like(root_ang_vel), max_size=num_history)
        self.last_actions = AutoQueue(torch.zeros_like(last_action), max_size=num_history)

    @torch.inference_mode()
    def compute(self, joint_pos, joint_vel, root_quat, root_ang_vel, last_action):
        self.joint_states.push(torch.cat([joint_pos, joint_vel * 0.05], dim=0))
        gravity = math_utils.quat_apply_inverse(root_quat, self.gravity_vec)
        self.gravities.push(gravity)
        self.root_ang_vels.push(root_ang_vel)
        self.last_actions.push(last_action)
        return dict(
            joint_states=self.joint_states.get_tensor(),
            gravities=self.gravities.get_tensor(),
            root_ang_vels=self.root_ang_vels.get_tensor(),
            last_actions=self.last_actions.get_tensor(),
            root_quat=root_quat,
        )

    @torch.inference_mode()
    def clear(self):
        self.joint_states.clear()
        self.gravities.clear()
        self.root_ang_vels.clear()
        self.last_actions.clear()

class G1Server:
    def __init__(self, args, device: Union[str, torch.device] = "cpu"):
        self.device = device
        self.env: Union[MujocoEnv, UnitreeEnv, UnitreeHeadEnv] = self._setup_env()
        # p_gains, d_gains to be set to real value
        self.collector = VlaDataCollector(data_dir='vla_data', task_desc='test task', frequency=50, p_gains=[0.5,0.5], d_gains=[0.1,0.1])
        if not self.collector.is_initialized:
            raise RuntimeError("[G1Server] VlaDataCollector failed to initialize. Camera validation failed.")
        time.sleep(1)
        self.policy = self._setup_policy()
        self.policy_started = False
        self.emergency_stop = False

        self.last_action = torch.zeros(len(JOINT_NAMES))
        self.gravity_vec = torch.zeros(3)
        self.root_quat = torch.zeros(4)
        self.ref_root_quat = torch.zeros(4)

        self.gravity_vec[2] = -1.0

        self.current_step = 0
        self.last_add_data_time = 0
        self.last_get_data_time = 0

        self.collect_frame_cnt = 0
        self.collect_start_root_quat = None
        
        
        self.intermediates = Intermediates(num_history=5, 
                                          joint_pos=torch.zeros(len(JOINT_NAMES)), 
                                          joint_vel=torch.zeros(len(JOINT_NAMES)), 
                                          gravity_vec=self.gravity_vec, 
                                          root_ang_vel=torch.zeros(3), 
                                          last_action=torch.zeros(len(JOINT_NAMES)))

        self.kp, self.kd = resolve_kpkd(
            hip_kp=120.0, hip_kd=3.0,
            knee_kp=160.0, knee_kd=4.0,
            ankle_kp=40.0, ankle_kd=1.0,
            waist_kp=200.0, waist_kd=5.0,
            shoulder_kp=40.0, shoulder_kd=1.0,
            arm_kp=40.0, arm_kd=1.0,
            wrist_kp=20.0, wrist_kd=0.5,
        )
        self.env.set_pd_gains(self.kp, self.kd)
        # self.motion_switcher = MotionSwitcher(motion_path="assets/lafan_motions_g1.pkl",
        #                                       urdf_path="assets/g1_29dof_mode_11.urdf",
        #                                       motion_quat_convention="wxyz",
        #                                       joint_names=JOINT_NAMES,
        #                                       eef_links=EEF_LINKS,
        #                                       switch_interval=5.0,
        #                                       num_futures=5,
        #                                       future_dt=0.02,
        #                                       initial_motion_id=2,
        #                                       initial_motion_time=0.0,
        #                                       motion_progress_bar=True)
        self.mocap_reciever = MocapReceiver(args, data_rate=50, num_history=5)
        self.mocap_reciever.launch()

        self.env.register_input_callback("gamepad.L1.pressed", self._start_callback)
        self.env.register_input_callback("_start", self._start_callback)
        self.env.register_input_callback("gamepad.L2.pressed", self._emergency_stop_callback)
        self.env.register_input_callback("_stop", self._emergency_stop_callback)
        self.env.register_input_callback("gamepad.R1.pressed", self._reset_history_callback)
        self.env.register_input_callback("r", self._reset_history_callback)
        self.env.register_input_callback("gamepad.R2.pressed", self._normalize_motion_callback)
        self.env.register_input_callback("_norm", self._normalize_motion_callback)
        # data collection button
        self.env.register_input_callback("gamepad.A.pressed", self._start_collect_callback)
        self.env.register_input_callback("gamepad.B.pressed", self._stop_collect_callback)
        # image show button
        # self.env.register_input_callback("gamepad.X.pressed", self._show_image)
        self.reset_lock = False
        self.collecting = False
        self.showing_image = False


        # inspire controller
        from deploy_utils.inspire_service import start_hand_service_w_state
        from multiprocessing.shared_memory import SharedMemory
        self.hand_qpos_shm = SharedMemory(create=True, size=12*4)
        self.hand_qpos = np.ndarray(12, dtype=np.float32, buffer=self.hand_qpos_shm.buf)

        self.left_hand_state_array = Array('d', 6, lock=True)
        self.right_hand_state_array = Array('d', 6, lock=True)
        hand_qpos_shm_name = self.hand_qpos_shm.name
        # set hand target and read hand states
        self.hand_service = Process(target=start_hand_service_w_state, args=(hand_qpos_shm_name, self.right_hand_state_array, self.left_hand_state_array))
        # self.hand_service.daemon = True
        self.hand_service.start()
        

    def _start_collect_callback(self):
        if not self.collecting:
            self.collecting = True
            self.collect_frame_cnt = 0
            state_tick = self.env.get_state_tick()
            self.collector.start_collection(state_msg_tick=state_tick)

    def _stop_collect_callback(self):
        if self.collecting:
            self.collecting = False
            self.collector.save_episode()
    
    # def _show_image(self):
    #     if not self.showing_image:
    #         self.showing_image = True
    #     else:
    #         self.showing_image = False
            

    def _start_callback(self):
        self.policy_started = True

    def _emergency_stop_callback(self):
        self.emergency_stop = True
        if self.collecting:
            self.collecting = False
            self.collector.save_episode()

    def _reset_history_callback(self):
        self.intermediates.clear()

    def _normalize_motion_callback(self):
        last_normalize_time = getattr(self, 'last_normalize_time', time.monotonic())
        if time.monotonic() - last_normalize_time > 1.0:
            yaw_root_quat = math_utils.yaw_quat(self.root_quat)
            yaw_ref_root_quat = math_utils.yaw_quat(self.ref_root_quat)
            delta_quat = math_utils.quat_mul(yaw_root_quat, math_utils.quat_conjugate(yaw_ref_root_quat))
            self.last_normalize_time = time.monotonic()
            self.mocap_reciever.set_delta_quat(delta_quat)
            print('Mocap Rotation Normalized.')
        else:
            self.last_normalize_time = time.monotonic()
            print('Mocap Rotation Not Normalized. Waiting for next normalization...')

    def _setup_env(self) -> Union[MujocoEnv, UnitreeEnv, UnitreeHeadEnv]:
        env = UnitreeHeadEnv(xml_path="assets/scene_29dof.xml",
                        joint_order=JOINT_NAMES,
                        action_joint_names=JOINT_NAMES,
                        control_freq=50,
                        simulation_freq=1000,
                        joint_damping=1.0,
                        joint_armature=0.01,
                        simulated_state=False,
                        joint_limits=JOINT_LIMITS,
                        torque_limits=TORQUE_LIMITS,
                        clip_action_to_torque_limit=False,
                        emergency_stop_breakpoint=True,
                        launch_input_thread=False,
                        emergency_stop_condition={
                            'joint_pos_limit': 0.98,
                            'ignore_limit_joints': ['left_ankle_pitch_joint', 'right_ankle_pitch_joint',
                                                    'left_hip_pitch_joint', 'right_hip_pitch_joint',
                                                    'left_hip_roll_joint', 'right_hip_roll_joint',
                                                    'left_knee_joint', 'right_knee_joint',
                                                    'left_wrist_roll_joint', 'right_wrist_roll_joint',
                                                    'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
                                                    'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
                                                    'left_elbow_joint', 'right_elbow_joint',
                                                    'waist_yaw_joint', 'waist_roll_joint',
                                                    'waist_pitch_joint'],
                            'roll_limit': 5.57,
                        },)
        if env.simulated:
            self.debug_spheres = [env.add_visual_sphere(pos=np.zeros(3), radius=0.05, rgba=(1, 1, 0, 1)) for _ in range(len(EEF_LINKS))]
        return env

    def _setup_policy(self) -> torch.jit.ScriptModule:
        policy = torch.jit.load("models/policy.pt")
        return policy.to(self.device).eval()

    @torch.inference_mode()
    def run_main_loop(self):
        action = torch.zeros(len(JOINT_NAMES))
        self.env.refresh_data()

        (
            joint_pos,
            joint_vel,
            joint_cmd,
            root_rpy,
            root_quat,
            root_ang_vel,
        ) = self.env.get_joint_root_data()

        initial_joint_pos = joint_pos
        initial_root_quat = root_quat
        self.root_quat[:] = initial_root_quat
        self.ref_root_quat[:] = initial_root_quat
        self._normalize_motion_callback()
        initial_joint_time = time.monotonic()

        mocap_data = self.mocap_reciever.get_data()
        while mocap_data is None:
            time.sleep(0.1)
            mocap_data = self.mocap_reciever.get_data()
            print('Waiting for mocap data...')

        mocap_body_names = mocap_data['body_names'][0]
        mocap_reindex = [mocap_body_names.index(name) for name in EEF_LINKS]

        while not self.policy_started:
            if self.env.simulated:
                self.env.set_root_fixed(True)
            while not self.env.step_complete():
                time.sleep(0.0001)
            self.env.refresh_data()

            (
                joint_pos,
                joint_vel,
                joint_cmd,
                root_rpy,
                root_quat,
                root_ang_vel,
            ) = self.env.get_joint_root_data()

            self.root_quat[:] = root_quat


            mocap_data = self.mocap_reciever.get_data()

            # get inspire data
            self.hand_qpos[:] = np.concatenate(
                [np.array(mocap_data['lhand_qpos_real'][-1]).astype(np.float32),
                np.array(mocap_data['rhand_qpos_real'][-1]).astype(np.float32),]
            ).reshape(-1)
            self.ref_root_quat[:] = mocap_data['body_quat'][0, mocap_reindex[0]]

            intermediates = self.intermediates.compute(joint_pos, 
                                                      joint_vel, 
                                                      root_quat, 
                                                      root_ang_vel, 
                                                      self.last_action)

            current_joint_time = time.monotonic()
            ratio = min((current_joint_time - initial_joint_time) / 5.0, 1.0)

            action[:] = initial_joint_pos * (1.0 - ratio)
            self.env.step(action, 0.0, 0.0)
            self.last_action = action.cpu().clone()

        all_actions = []
        all_joint_pos = []
        all_joint_vel = []
        if self.env.simulated:
            self.env.set_root_fixed(False)
        start_policy_time = time.monotonic()
        loop_count = 0
        # terminal plot
        env_get_data_times = []
        mocap_get_data_times = []
        obs_times = []
        policy_times = []
        add_data_times = []

        while True:
            loop_count += 1
            while not self.env.step_complete():
                time.sleep(0.0001)
            
            # all the get data functions should be no-wait mode, should not block this loop
            # start_time = time.monotonic()
            self.env.refresh_data()
            (
                joint_pos,
                joint_vel,
                joint_cmd,
                root_rpy,
                root_quat,
                root_ang_vel,
            ) = self.env.get_joint_root_data()

            self.root_quat[:] = root_quat

            state_tick = self.env.get_state_tick()

            # for state collection, this two values should be radians
            pitch_state, yaw_state = self.env.get_head_state()
            pitch_state_rad = pitch_state * np.pi / 180.0
            yaw_state_rad = yaw_state * np.pi / 180.0

            # env_get_data_time = time.monotonic()

            mocap_data = self.mocap_reciever.get_data()
            motion_rb_pos = mocap_data['body_pos'][:, mocap_reindex]
            motion_rb_quat = mocap_data['body_quat'][:, mocap_reindex]
            motion_rb_lin_vel = mocap_data['body_lin_vel'][:, mocap_reindex]
            self.ref_root_quat[:] = mocap_data['body_quat'][0, mocap_reindex[0]]

            motion_qpos = mocap_data['qpos'][-1]
            
            self.hand_qpos[:6] = np.array(mocap_data['lhand_qpos_real'][-1]).astype(float)
            self.hand_qpos[6:] = np.array(mocap_data['rhand_qpos_real'][-1]).astype(float)

            hip_rot = mocap_data['hip_rot']
            head_rot = mocap_data['head_rot']
            # in degree format
            relative_pitch, relative_yaw = get_relative_head_angles(head_rot[-1], hip_rot[-1])
            tgt_pitch_rad = relative_pitch * np.pi / 180.0
            tgt_yaw_rad = relative_yaw * np.pi / 180.0 
            
            # mocap_get_data_time = time.monotonic()

            if self.emergency_stop:
                if self.env.simulated:
                    self.env.set_root_fixed(True)
                print('Emergency stopped')
                self.env.step(joint_pos, 0., 0.)
                # torch.save(all_actions, "all_actions.pt")
                # torch.save(all_joint_pos, "all_joint_pos.pt")
                # torch.save(all_joint_vel, "all_joint_vel.pt")
                break

            if self.env.simulated:
                for i, sphere in enumerate(self.debug_spheres):
                    sphere_pos = motion_rb_pos[0, i].cpu().numpy().copy()
                    sphere.pos = sphere_pos

            intermediates = self.intermediates.compute(joint_pos, 
                                                      joint_vel, 
                                                      root_quat, 
                                                      root_ang_vel, 
                                                      self.last_action)
            observation = compute_observation(**intermediates,
                                              motion_rb_pos=motion_rb_pos, 
                                              motion_rb_quat=motion_rb_quat,  
                                              root_lin_vel=motion_rb_lin_vel[:, 0])
            # compute_obs_time = time.monotonic()

            for k, v in observation.items():
                observation[k] = v.to(self.device).unsqueeze(0)
            action = self.policy(**observation)[0].clamp(-10.0, 10.0)
            all_actions.append(action.cpu().clone())
            all_joint_pos.append(joint_pos.cpu().clone())
            all_joint_vel.append(joint_vel.cpu().clone())
            self.last_action = action.cpu().clone()
            policy_inference_time = time.monotonic()
            if loop_count > 1000:
                self.env.step(action, 38-relative_pitch, relative_yaw)
            else:
                action[:] = initial_joint_pos * 0
                self.env.step(action, 0.0, 0.0)

            states = {
                "left_arm": {                                                                    
                    "qpos":   joint_pos[15: 22],
                    "qvel":   joint_vel[15: 22],
                    "torque": [],
                }, 
                "right_arm": {                                                                    
                    "qpos":   joint_pos[22: 29],
                    "qvel":   joint_vel[22: 29],                         
                    "torque": [],                         
                },                        
                "left_hand": {                                                                    
                    "qpos":   list(self.left_hand_state_array),
                    "qvel":   [],                           
                    "torque": [],                          
                }, 
                "right_hand": {                                                                    
                    "qpos":   list(self.right_hand_state_array),
                    "qvel":   [],                           
                    "torque": [],  
                }, 
                "waist": {
                    "qpos":   joint_pos[12: 15],
                    "qvel":   joint_vel[12: 15],
                    "torque": [],
                },
                "left_leg": {
                    "qpos":   joint_pos[0: 6],
                    "qvel":   joint_vel[0: 6],
                    "torque": [],
                },
                "right_leg": {
                    "qpos":   joint_pos[6: 12],
                    "qvel":   joint_vel[6: 12],
                    "torque": [],
                },
                "imu": {
                    "ang_vel": root_ang_vel,
                    "quat": root_quat,
                }, 
                "head_servo": {
                    "qpos": (pitch_state_rad, yaw_state_rad)
                },
                
            }
            actions = {
                "left_arm": {                                                                    
                    "qpos":   action[15: 22], # target dof_pos = scaled_actions + default_pos
                    "qvel":   [],
                    "torque": [], # torque calculated = (tgt_pos - dof_pos) * p_gains - dof_vel * d_gains
                }, 
                "right_arm": {                                                                    
                    "qpos":   action[22: 29],       
                    "qvel":   [],                          
                    "torque": [],                         
                },                        
                "left_hand": {                                                                    
                    "qpos":   self.hand_qpos[:6],           
                    "qvel":   [],                           
                    "torque": [],                          
                }, 
                "right_hand": {                                                                    
                    "qpos":   self.hand_qpos[6:],       
                    "qvel":   [],                           
                    "torque": [],  
                }, 
                "waist": {
                    "qpos":   action[12: 15],
                    "qvel":   [],
                    "torque": [],
                },
                "left_leg": {
                    "qpos":   action[0: 6],
                    "qvel":   [],
                    "torque": [],
                },
                "right_leg": {
                    "qpos":   action[6: 12],
                    "qvel":   [],
                    "torque": [],
                },
                "head_servo": {
                    "qpos": (tgt_pitch_rad, tgt_yaw_rad)
                },
            }

            
            # task_obs = {
            #     "mocap_body_pos": motion_rb_pos,
            #     "mocap_body_quat": motion_rb_quat,
            #     "mocap_root_lin_vel": motion_rb_lin_vel[:, 0],
            # }
            if self.collecting:
                if self.collect_frame_cnt == 0:
                    # start collecting button pressed, record the start root quat
                    self.collect_start_root_quat = motion_rb_quat[-1, 0].clone()

                current_root_quat = motion_rb_quat[-1, 0].clone()
                relative_root_quat = get_relative_root_quat(current_root_quat, self.collect_start_root_quat)

                task_obs = {
                    "mocap_body_pos": motion_rb_pos,
                    "mocap_body_quat": motion_rb_quat,
                    "mocap_root_lin_vel": motion_rb_lin_vel[:, 0],
                    "mocap_qpos": motion_qpos,
                    "mocap_relative_root_quat": relative_root_quat,
                }
                # If frame loss for 1 second, quit collecting
                collect_flag = self.collector.add_data_to_episode(
                    states=states,
                    actions=actions,
                    task_obs=task_obs,
                    state_msg_tick=state_tick
                )
                self.collect_frame_cnt += 1
                if not collect_flag:
                    print('Data collect process died, please restart the script to continue collecting')
                    self.collecting = False
            add_data_time = time.monotonic()
            # if self.showing_image:
            #     self.collector.show_image()
            
            if time.monotonic() - start_policy_time > 1.0:
                print(f"Policy Frequency:", self.env.step_frequency)
                if self.env.step_frequency < 45:
                    self.last_step = self.current_step
                    self.current_step = time.time()
                    step_interval = self.current_step - self.last_step
                    print(f'frame loss interval: {step_interval}')
                start_policy_time = time.monotonic()
                

            self.last_add_data_time = add_data_time - policy_inference_time


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7002)
    
    server = G1Server(args=parser.parse_args(), device="cuda")
    server.run_main_loop()
