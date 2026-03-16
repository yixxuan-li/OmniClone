from mujoco.viewer import np
import argparse
from multiprocessing import Process, Array
from multiprocessing.shared_memory import SharedMemory
import torch
import time
from tqdm import trange
import math
import socket
import json
import copy
import cv2



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
from deploy_utils.inspire_service import InspireController
from utils.utils import get_relative_head_angles, ControlDataManager
from server import compute_observation, Intermediates
from server_head import G1_Head_Server
from utils.vla_base_client import RobotInferenceClient
from general_motion_retargeting import RobotMotionViewer

# inspire controller
from utils.test_hand import start_hand_service_w_state

from config import IMG_CONFIG, INIT_STATE
import matplotlib.pyplot as plt
import os


def visualize_actions(pred_dict, gt_dict, save_path: str = "./vla_action_visualization.png"):
    """
    Plots ground truth vs predicted actions for each joint/dimension.
    """
    # 1. Prepare data
    keys = list(gt_dict.keys())
    num_vars = len(keys)
    
    # Create a figure with subplots for each category
    fig, axes = plt.subplots(num_vars, 1, figsize=(10, 3 * num_vars), constrained_layout=True)
    
    # Handle the case where there is only one key (axes won't be a list)
    if num_vars == 1:
        axes = [axes]

    for i, key in enumerate(keys):
        # Shapes are (1, 30, act_dims) -> remove batch dim to get (30, act_dims)
        gt_data = np.squeeze(gt_dict[key], axis=0) 
        pred_data = np.squeeze(pred_dict[key], axis=0)
        
        timesteps = np.arange(pred_dict.shape[0]) # 0 to 29
        num_dims = gt_data.shape[1] # number of joints in this group

        for d in range(num_dims):
            color = plt.cm.tab10(d % 10) # Use consistent colors for same joints
            
            # Plot GT as dashed lines, Pred as solid lines
            axes[i].plot(timesteps, gt_data[:, d], label=f'Dim {d} GT', 
                         linestyle='--', color=color, alpha=0.6)
            axes[i].plot(timesteps, pred_data[:, d], label=f'Dim {d} Pred', 
                         linestyle='-', color=color, linewidth=2)

        axes[i].set_title(f'Trajectory Comparison: {key}')
        axes[i].set_xlabel('Time Step (Chunk Index)')
        axes[i].set_ylabel('Value')
        
        # Put legend to the side if there are many dimensions
        axes[i].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=2)
        axes[i].grid(True, alpha=0.3)

    plt.savefig(save_path, bbox_inches='tight')
    print(f"Trajectory visualization saved to {save_path}")
    plt.close()

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


class G1_Vla_Server(G1_Head_Server):
    def __init__(self, args, device: str | torch.device = "cpu"):
        self.vlm_server_ip = args.vlm_server_ip
        self.vlm_server_port = args.vlm_server_port
        self.device = device
        self.debug = args.debug
        self.replay = args.replay
        self.replay_state = args.replay_state
        self.record = args.record
        self.episode_length = args_cli.episode_length
        self.action_horizon = args.action_horizon
        self.robot_type = args.robot_type
        self.task = args.task

        super().__init__(args, device)

        self.head_pitch_state_rad = 0.0
        self.head_yaw_state_rad = 0.0


        self.control_data_manager = ControlDataManager(args=args,
                                              data_rate=50,
                                              num_history=5)
        self.viewer = RobotMotionViewer(
            robot_type=self.robot_type,
            transparent_robot=1,
            motion_fps=1000
        )


    def _normalize_motion_callback(self):
        last_normalize_time = getattr(self, 'last_normalize_time', time.monotonic())
        if time.monotonic() - last_normalize_time > 1.0:
            yaw_root_quat = math_utils.yaw_quat(self.root_quat)
            yaw_ref_root_quat = math_utils.yaw_quat(self.ref_root_quat)
            delta_quat = math_utils.quat_mul(yaw_root_quat, math_utils.quat_conjugate(yaw_ref_root_quat))
            self.last_normalize_time = time.monotonic()
            self.control_data_manager.set_delta_quat(delta_quat)
            print('Mocap Rotation Normalized.')
        else:
            self.last_normalize_time = time.monotonic()
            print('Mocap Rotation Not Normalized. Waiting for next normalization...')

    def _prepare_ctrl_scource(self):
        if self.debug or self.replay:
            self.vla_client = None
        else:
            self.vla_client: RobotInferenceClient = RobotInferenceClient(
                                                                        host=self.vlm_server_ip,
                                                                        port=self.vlm_server_port,
                                                                        )
            print(f"[INFO] Connecting to policy server at {self.vlm_server_ip}:{self.vlm_server_port}")

    def _get_vla_observation(self):
        obs = {}
        # get head image 
        head_image = self.collector.img_array.copy()
        cv2.imwrite("debug_head_image.png", head_image)
        head_image = head_image[:,:,[2, 1, 0]]
        # resized_head_image = cv2.resize(head_image, (IMG_CONFIG['tv_img_shape'][1] // 2, IMG_CONFIG['tv_img_shape'][0] // 2))
        obs["video.ego_view"] = np.expand_dims(np.expand_dims(head_image, axis=0), axis=0)

        # get joint data
        joint_data = self.env.get_joint_data()
        root_data = self.env.get_root_data()

        root_ang_vel = root_data['root_ang_vel']
        root_quat = root_data['root_quat']
        joint_pos = joint_data['joint_pos']
        joint_vel = joint_data['joint_vel']
        right_arm_pos = joint_pos[22:29].detach().clone().to(self.device)
        # right_arm_pos[0] += 0.35
        # right_arm_pos[5] -=0.40
        # print(f'self.left_hand_state_array: {self.left_hand_state_array}')
        obs["state.state"] = torch.cat([
                                        joint_pos[15:22].detach().clone().to(self.device),    # left arm pos
                                        joint_vel[15:22].detach().clone().to(self.device),     # left arm vel
                                        # right_arm_pos,
                                        joint_pos[22:29].detach().clone().to(self.device),    # right arm pos
                                        joint_vel[22:29].detach().clone().to(self.device),     # right arm vel
                                        torch.tensor(self.left_hand_state_array).to(self.device),
                                        torch.tensor(self.right_hand_state_array).to(self.device),
                                        joint_pos[12:15].detach().clone().to(self.device),    # waist
                                        joint_vel[12:15].detach().clone().to(self.device),    # waist
                                        joint_pos[0:6].detach().clone().to(self.device),    # left leg
                                        joint_vel[0:6].detach().clone().to(self.device),    # left leg
                                        joint_pos[6:12].detach().clone().to(self.device),    # right leg
                                        joint_vel[6:12].detach().clone().to(self.device),    # right leg
                                        root_ang_vel.detach().clone().to(self.device),    # root ang vel
                                        # root_quat.detach().clone().to(self.device),    # root quat
                                        torch.tensor([0.96, -0.0079, 0.0247, 0.275]).to(self.device),    # root quat
                                        # torch.tensor([0.96, -0.0079, 0.0247, 0.275]).to(self.device),    # root quat
                                        torch.tensor([self.head_pitch_state_rad, self.head_yaw_state_rad]).to(self.device),
                                        # torch.tensor([0.0, 0.0]).to(self.device)
                                    ], dim=-1).unsqueeze(0).unsqueeze(0).cpu().numpy()
        obs["annotation.human.action.task_description"] = [self.task]

        return obs

    def _get_vla_data_state(self, episode_step: int):
        # replay state
        obs = {}
        # resized_head_image = cv2.resize(head_image, (IMG_CONFIG['tv_img_shape'][1] // 2, IMG_CONFIG['tv_img_shape'][0] // 2))
        img_path = "/home/whx/data/251217/episode_0042/" + self.data[episode_step]['colors']['color_0']
        head_image = cv2.imread(img_path)
        head_image = head_image[:,:,[2, 1, 0]]
        obs["video.ego_view"] = np.expand_dims(np.expand_dims(head_image, axis=0), axis=0)

        # head_image = self.collector.img_array.copy()
        # cv2.imwrite("debug_head_image.png", head_image)
        # head_image = head_image[:,:,[2, 1, 0]]
        # # resized_head_image = cv2.resize(head_image, (IMG_CONFIG['tv_img_shape'][1] // 2, IMG_CONFIG['tv_img_shape'][0] // 2))
        # obs["video.ego_view"] = np.expand_dims(np.expand_dims(head_image, axis=0), axis=0)
        
        # # get joint data
        # joint_data = self.env.get_joint_data()
        # root_data = self.env.get_root_data()

        # root_ang_vel = root_data['root_ang_vel']
        # root_quat = root_data['root_quat']
        # joint_pos = joint_data['joint_pos']
        # joint_vel = joint_data['joint_vel']
        # right_arm_pos = joint_pos[22:29].detach().clone().to(self.device)
        # # right_arm_pos[0] += 0.35
        # # right_arm_pos[5] -=0.40
        # print(f'self.left_hand_state_array: {self.left_hand_state_array}')
        # obs["state.state"] = torch.cat([
        #                                 joint_pos[15:22].detach().clone().to(self.device),    # left arm pos
        #                                 joint_vel[15:22].detach().clone().to(self.device),     # left arm vel
        #                                 # right_arm_pos,
        #                                 joint_pos[22:29].detach().clone().to(self.device),    # right arm pos
        #                                 joint_vel[22:29].detach().clone().to(self.device),     # right arm vel
        #                                 torch.tensor(self.left_hand_state_array).to(self.device),
        #                                 torch.tensor(self.right_hand_state_array).to(self.device),
        #                                 joint_pos[12:15].detach().clone().to(self.device),    # waist
        #                                 joint_vel[12:15].detach().clone().to(self.device),    # waist
        #                                 joint_pos[0:6].detach().clone().to(self.device),    # left leg
        #                                 joint_vel[0:6].detach().clone().to(self.device),    # left leg
        #                                 joint_pos[6:12].detach().clone().to(self.device),    # right leg
        #                                 joint_vel[6:12].detach().clone().to(self.device),    # right leg
        #                                 root_ang_vel.detach().clone().to(self.device),    # root ang vel
        #                                 root_quat.detach().clone().to(self.device),    # root quat
        #                                 # torch.tensor([0.96, -0.0079, 0.0247, 0.275]).to(self.device),    # root quat
        #                                 # torch.tensor([self.head_pitch_state_rad, self.head_yaw_state_rad]).to(self.device),
        #                                 torch.tensor([0.0, 0.0]).to(self.device)
        #                             ], dim=-1).unsqueeze(0).unsqueeze(0).cpu().numpy()
        # obs["annotation.human.action.task_description"] = [self.task]


        obs["state.state"] = torch.cat([
                                        torch.tensor(self.data[episode_step]['states']['left_arm']['qpos']).to(self.device),    # left arm pos
                                        torch.tensor(self.data[episode_step]['states']['left_arm']['qvel']).to(self.device),    # left arm vel
                                        torch.tensor(self.data[episode_step]['states']['right_arm']['qpos']).to(self.device),   # right arm pos
                                        torch.tensor(self.data[episode_step]['states']['right_arm']['qvel']).to(self.device),   # right arm vel
                                        torch.tensor(self.data[episode_step]['states']['left_hand']['qpos']).to(self.device),
                                        torch.tensor(self.data[episode_step]['states']['right_hand']['qpos']).to(self.device),
                                        torch.tensor(self.data[episode_step]['states']['waist']['qpos']).to(self.device),
                                        torch.tensor(self.data[episode_step]['states']['waist']['qvel']).to(self.device),
                                        torch.tensor(self.data[episode_step]['states']['left_leg']['qpos']).to(self.device),
                                        torch.tensor(self.data[episode_step]['states']['left_leg']['qvel']).to(self.device),
                                        torch.tensor(self.data[episode_step]['states']['right_leg']['qpos']).to(self.device),
                                        torch.tensor(self.data[episode_step]['states']['right_leg']['qvel']).to(self.device),
                                        torch.tensor(self.data[episode_step]['states']['imu']['ang_vel']).to(self.device),    # root ang vel
                                        torch.tensor(self.data[episode_step]['states']['imu']['quat']).to(self.device),    # root quat
                                        torch.tensor(self.data[episode_step]['states']['head_servo']['qpos']).to(self.device)
                                    ], dim=-1).unsqueeze(0).unsqueeze(0).cpu().numpy()
        obs["annotation.human.action.task_description"] = [self.task]

        raw_actions =  self.data[episode_step]['actions']
        gt_action = {
                    'action.left_arm_qpos': np.expand_dims(raw_actions['left_arm']['qpos'], axis=0),
                    'action.right_arm_qpos': np.expand_dims(raw_actions['right_arm']['qpos'], axis=0),
                    'action.left_hand_joint': np.expand_dims(raw_actions['left_hand']['qpos'], axis=0),
                    'action.right_hand_joint': np.expand_dims(raw_actions['right_hand']['qpos'], axis=0),
                    'action.head_pose': np.expand_dims(raw_actions['head_servo']['qpos'], axis=0),
                    'action.pelvis_xyz': np.zeros((1, 30, 3)),
                    'action.root_xy_velocity': np.zeros((1, 30, 2)),
                    'action.waist': np.expand_dims(raw_actions['waist']['qpos'], axis=0)
        }
        return obs, gt_action

    def _parse_vla_observations(self):
        obs_dict = self._get_vla_observation()

        # parse all observations for VLM
        obs = copy.deepcopy(obs_dict)
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # preprocess the head camera image
        image = torch.squeeze(obs_dict["head_cam"]).to(torch.uint8)
        image = image.permute(1, 2, 0)
        # BGR to RGB
        image = image[:,:,[2, 1, 0]]     # image = image / 255.0

        # cv2.imshow("image", image)
        obs["head_cam"] = image


        obs_dict = {
            "video.ego_view": obs["head_cam"].detach().cpu().numpy()[np.newaxis, :, :, :],
            # "state.joint_position": joint_positions[np.newaxis, :].astype(np.float64),
            "state.state": obs["robot_joint_states"].detach().cpu().numpy()[np.newaxis, 15:],
            "annotation.human.action.task_description": ["pick up the pink toy rabbit and put to the blue bowl"], #pick up the green tomato and put to the basket
        }
        
        return obs_dict

    def _parse_vla_actions(self, obs_dict):
        actions = {
            'action.left_arm_qpos': obs_dict['action.left_arm'],
            'action.right_arm_qpos': obs_dict['action.right_arm'],
            'action.left_hand_joint': obs_dict['action.left_hand'],
            'action.right_hand_joint': obs_dict['action.right_hand'],
            'action.lleg_qpos': obs_dict['action.left_leg'],
            'action.rleg_qpos': obs_dict['action.right_leg'],
            'action.head_pose': obs_dict['action.head'],
            # 'action.pelvis_xyz': obs_dict['action.pelvis_pos'],
            'action.root_xy_velocity': obs_dict['action.pelvis_vel'],
            'action.waist': obs_dict['action.waist'],
            'action.rel_quat':obs_dict['action.body_rel_quat'],
            'action.abs_quat': obs_dict['action.body_quat']
        }
        # for k, v in actions.items():
        #     print(f"----------------{k}'s action shape: {v.shape}----------------")

        return actions


    def _vla_chunk_step(self, action, chunk_step):
        action = action
        full_qpos = self.viewer.data.qpos.copy()
        # import pdb; pdb.set_trace()
        # full_qpos[:3] = action['action.pelvis_xyz'][0][chunk_step] + 0.6
        # print(action['action.abs_quat'][0][chunk_step])
        full_qpos[3:7] = action['action.abs_quat'][0][chunk_step]
        full_qpos[7+12:7+12+3] = action['action.waist'][0][chunk_step]
        full_qpos[7+0:7+0+6] = action['action.lleg_qpos'][0][chunk_step]
        full_qpos[7+6:7+6+6] = action['action.rleg_qpos'][0][chunk_step]
        full_qpos[7+15:7+15+14] = np.concatenate([action['action.left_arm_qpos'][0][chunk_step], action['action.right_arm_qpos'][0][chunk_step]], axis=-1)

        self.viewer.step(
            root_pos=full_qpos[:3],
            root_rot=full_qpos[3:7],
            dof_pos=full_qpos[7:],
        )

        chunk_data = self.viewer.get_ref_body_data()
        chunk_data['lhand_qpos_real'] = action['action.left_hand_joint'][0][chunk_step]
        chunk_data['rhand_qpos_real'] = action['action.right_hand_joint'][0][chunk_step]
        chunk_data['head_pose'] = action['action.head_pose'][0][chunk_step] / math.pi * 180.0
        chunk_data['root_xy_velocity'] = action['action.root_xy_velocity'][0][chunk_step]

        return chunk_data

    def _vla_episode_infer(self, debug: bool = False, episode_step: int = 0):
        if not debug:
            if self.replay_state:
                obs_dict, gt_action = self._get_vla_data_state(episode_step)
            else:
                obs_dict = self._get_vla_observation()

            # print("********************", obs_dict['video.ego_view'].shape, "********************")
            # print("====================", obs_dict['state.state'].shape, "====================")
            # print("state head pose: ", obs_dict["state.state"][:, :, -2:])
            raw_actions, _ = self.vla_client.get_action(obs_dict)
            # print("********************", raw_actions.keys(), "********************")
            raw_actions = self._parse_vla_actions(raw_actions)
            # print("action head pose: ", raw_actions['action.head_pose'][0])
            raw_actions['state.imu_quat'] = obs_dict['state.state'][:,:, -6:-2].repeat(self.action_horizon, axis=1)
            # if self.replay_state:
            #     visualize_actions(raw_actions, gt_action)
        else:
            raw_actions = {
                # 'action.left_arm_qpos': np.zeros((1, self.action_horizon, 7)),
                'action.left_arm_qpos': np.expand_dims(np.array([0.15920503527514324, 0.4252859092217435, -0.04942028258704028, 0.26571326624491803, -0.2113188466820098, -0.5237687916402002, 0.07444325362920912]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
                # 'action.right_arm_qpos': np.zeros((1, self.action_horizon, 7)),
                'action.right_arm_qpos': np.expand_dims(np.array([0.14540075995836405, -0.424680938568589, 0.11795725332533448, 0.09036890542637277, 0.2642860405774068, -0.49822768255898536, 0.21566704710968165]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
                'action.left_hand_joint': np.ones((1, self.action_horizon, 6)),
                'action.right_hand_joint': np.ones((1, self.action_horizon, 6)),
                # 'action.head_pose': np.zeros((1, self.action_horizon, 2)),
                'action.head_pose': np.expand_dims(np.array([0.7323535843623536, -0.06607623347560303]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
                # 'action.pelvis_xyz': np.zeros((1, self.action_horizon, 3)),
                'action.pelvis_xyz': np.expand_dims(np.array([0.45630666613578796, 0.5740872025489807, 0.8052271604537964]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
                # 'action.root_xy_velocity': np.zeros((1, self.action_horizon, 2)),
                'action.root_xy_velocity': np.expand_dims(np.array([0.002022087574005127, 0.005882978439331055]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
                # 'action.waist': np.zeros((1, self.action_horizon, 3)),
                'action.waist': np.expand_dims(np.array([0.049568961697862804, -0.016085483426647867, 0.1762881347282114]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
                # 'action.lleg_qpos': np.zeros((1, self.action_horizon, 6)),
                'action.lleg_qpos': np.expand_dims(np.array([0.3941409646420341, 0.040252663926680796, 0.07258134604202592, -0.08654551717956825, -0.09005026800268968, -0.06702218509924837]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
                # 'action.rleg_qpos': np.zeros((1, self.action_horizon, 6)),
                'action.rleg_qpos': np.expand_dims(np.array([0.3831512243844021, -0.13187685881076283, -0.10568242792783168, -0.08629745939279894, -0.047944851104199174, 0.16641061832269075]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
                'action.abs_quat': np.expand_dims(np.array([0.9938714504241943, 0.01961645483970642, -0.07925314456224442, 0.0745229721069336]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
                # 'state.imu_quat': np.expand_dims(np.array([1, 0, 0, 0]), axis=0).repeat(self.action_horizon, axis=0)[np.newaxis, :, :],
            }
        return raw_actions

    @torch.inference_mode()
    def run_main_loop(self):
        action = torch.zeros(len(JOINT_NAMES))
        self.env.refresh_data()
        joint_data = self.env.get_joint_data()
        root_data = self.env.get_root_data()
        # initial_joint_pos = joint_data['joint_pos']
        initial_joint_pos_ = INIT_STATE['joint_pos']
        initial_joint_pos = torch.tensor(initial_joint_pos_).float()
        # initial_root_quat = torch.tensor(INIT_STATE['quat']).float()
        initial_root_quat = root_data['root_quat']
        self.root_quat[:] = initial_root_quat
        self.ref_root_quat[:] = torch.tensor(INIT_STATE['quat']).float()
        self._normalize_motion_callback()
        initial_joint_time = time.monotonic()
        episode_freq = 0

        control_chunk_data = self._vla_episode_infer(debug=True)
        control_data = self._vla_chunk_step(control_chunk_data, chunk_step = 0)
        self.control_data_manager.receive_data(control_data)
        while control_chunk_data is None:
            time.sleep(0.1)
            control_chunk_data = self._vla_episode_infer(debug=True)
            control_data = self._vla_chunk_step(control_chunk_data, chunk_step = 0)
            self.control_data_manager.receive_data(control_data)
            print('Waiting for control signal...')


        vla_control_body_names = control_data['body_names']
        vla_control_reindex = [vla_control_body_names.index(name) for name in EEF_LINKS]

        while not self.policy_started:
            if self.env.simulated:
                self.env.set_root_fixed(True)
            while not self.env.step_complete():
                time.sleep(0.0001)
            self.env.refresh_data()
            joint_data = self.env.get_joint_data()
            root_data = self.env.get_root_data()
            self.root_quat[:] = root_data['root_quat']
            head_pitch_state, head_yaw_state = self.env.get_head_state()
            self.head_pitch_state_rad = head_pitch_state * np.pi / 180.0
            self.head_yaw_state_rad = head_yaw_state * np.pi / 180.0

            control_chunk_data = self._vla_episode_infer(debug = True)
            control_data = self._vla_chunk_step(control_chunk_data, 0)
            self.control_data_manager.receive_data(control_data)

            self.hand_qpos[:] = np.concatenate(
                [np.array(control_data['lhand_qpos_real']).astype(np.float32),
                np.array(control_data['rhand_qpos_real']).astype(np.float32),]
            ).reshape(-1)
            managed_control_data = self.control_data_manager.get_data()
            self.ref_root_quat[:] = managed_control_data['body_quat'][0, vla_control_reindex[0]]

            intermediates = self.intermediates.compute(joint_data['joint_pos'], 
                                                      joint_data['joint_vel'], 
                                                      root_data['root_quat'], 
                                                      root_data['root_ang_vel'], 
                                                      self.last_action)

            current_joint_time = time.monotonic()
            ratio = max(min(1 - (current_joint_time - initial_joint_time) / 5.0, 1.0), 0.0)
            action[:] = initial_joint_pos * (1.0 - ratio)
            self.env.step(action, 0.0, 0.0)
            self.last_action = action.cpu().clone()

        all_actions = []
        all_joint_pos = []
        all_joint_vel = []
        if self.env.simulated:
            self.env.set_root_fixed(False)
        start_policy_time = time.monotonic()

        if self.replay or self.replay_state:
            data_path = "/home/lyx/0108/episode_0066/data.json"
            # "/home/whx/Downloads/act_pred.json"
            # data_path = "/home/whx/data/251217/episode_0041/data.json"
            data_file = json.load(open(data_path, 'r'))
            self.data = data_file['data']
        for episode_step in range(self.episode_length):
            # print(f"\n=== Episode Step: {episode_step} ===")
            episode_start_timer = time.time()
            if self.replay:
                # raw_actions = self.data[episode_step]['actions']
                # raw_states = self.data[episode_step]['states']
                raw_actions_left_arm = self.data[episode_step]['task_obs']['mocap_qpos'][22:29]
                raw_actions_right_arm = self.data[episode_step]['task_obs']['mocap_qpos'][29:36]
                raw_actions_left_leg = self.data[episode_step]['task_obs']['mocap_qpos'][7:13]
                raw_actions_right_leg = self.data[episode_step]['task_obs']['mocap_qpos'][13:19]
                raw_actions_waist = self.data[episode_step]['task_obs']['mocap_qpos'][19:22]
                raw_actions_left_hand = self.data[episode_step]['actions']['left_hand']['qpos']
                raw_actions_right_hand = self.data[episode_step]['actions']['right_hand']['qpos']
                raw_actions_head = self.data[episode_step]['actions']['head_servo']['qpos']
                raw_actions_root_xyz_vel= self.data[episode_step]['task_obs']['mocap_root_lin_vel'][0]
                raw_actions_body_rel_quat = self.data[episode_step]['task_obs']['mocap_relative_root_quat'][0]
                raw_actions_root_quat = self.data[episode_step]['task_obs']['mocap_body_quat'][0][0]

                raw_actions = {
                    'action.left_arm_qpos': np.expand_dims(np.expand_dims(raw_actions_left_arm, axis=0), axis=0),
                    'action.right_arm_qpos': np.expand_dims(np.expand_dims(raw_actions_right_arm, axis=0), axis=0),
                    'action.lleg_qpos': np.expand_dims(np.expand_dims(raw_actions_left_leg, axis=0), axis=0),
                    'action.rleg_qpos': np.expand_dims(np.expand_dims(raw_actions_right_leg, axis=0), axis=0),
                    'action.left_hand_joint': np.expand_dims(np.expand_dims(raw_actions_left_hand, axis=0), axis=0),
                    'action.right_hand_joint': np.expand_dims(np.expand_dims(raw_actions_right_hand, axis=0), axis=0),
                    'action.head_pose': np.expand_dims(np.expand_dims(raw_actions_head, axis=0), axis=0),
                    'action.pelvis_xyz': np.zeros((1, 1, 3)),
                    'action.root_xy_velocity': np.expand_dims(np.expand_dims(raw_actions_root_xyz_vel, axis=0), axis=0), #np.zeros((1, 1, 2)),
                    'action.waist': np.expand_dims(np.expand_dims(raw_actions_waist, axis=0), axis=0),
                    'action.rel_quat': np.expand_dims(np.expand_dims(raw_actions_body_rel_quat, axis=0), axis=0),
                    'action.abs_quat': np.expand_dims(np.expand_dims(raw_actions_root_quat, axis=0), axis=0)
                    # 'action.lleg_qpos': np.expand_dims(np.expand_dims(raw_actions['left_leg']['qpos'], axis=0), axis=0),
                    # 'action.rleg_qpos': np.expand_dims(np.expand_dims(raw_actions['right_leg']['qpos'], axis=0), axis=0),
                    # 'state.imu_quat': np.expand_dims(np.expand_dims(raw_states['imu']['quat'], axis=0), axis=0)
                }
                control_chunk_data = raw_actions
                # control_chunk_data = self._vla_episode_infer(debug=True, episode_step=episode_step*8)
                self.action_horizon = 1
            else:
                if self.replay_state:
                    self.action_horizon = 8
                control_chunk_data = self._vla_episode_infer(debug=self.args.debug, episode_step=episode_step*8)

            for chunk_step in range(self.action_horizon):
                while not self.env.step_complete():
                    time.sleep(0.0001)
                control_data = self._vla_chunk_step(control_chunk_data, chunk_step)
                self.control_data_manager.receive_data(control_data)
                self.env.refresh_data()
                joint_data = self.env.get_joint_data()
                root_data = self.env.get_root_data()

                self.root_quat[:] = root_data['root_quat']

                managed_control_data = self.control_data_manager.get_data()
                motion_rb_pos = managed_control_data['body_pos'][:, vla_control_reindex]
                motion_rb_quat = managed_control_data['body_quat'][:, vla_control_reindex]
                motion_rb_lin_vel = managed_control_data['body_lin_vel'][:, vla_control_reindex]
                self.ref_root_quat[:] = managed_control_data['body_quat'][0, vla_control_reindex[0]]

                head_pitch_state, head_yaw_state = self.env.get_head_state()
                self.head_pitch_state_rad = head_pitch_state * np.pi / 180.0
                self.head_yaw_state_rad = head_yaw_state * np.pi / 180.0
                
                self.hand_qpos[:6] = np.array(control_data['lhand_qpos_real'][-1]).astype(float)
                self.hand_qpos[6:] = np.array(control_data['rhand_qpos_real'][-1]).astype(float)



                # in degree format
                # tgt_head_pitch_rad, tgt_head_yaw_rad, tgt_head_pitch_deg, tgt_head_yaw_deg = get_relative_head_angles(control_data['head_pose'][0], control_data['head_pose'][1])
                tgt_head_pitch_deg, tgt_head_yaw_deg = control_data['head_pose'][0], control_data['head_pose'][1]
                tgt_head_pitch_rad = tgt_head_pitch_deg / 180. * 3.14
                tgt_head_yaw_rad = tgt_head_yaw_deg / 180. * 3.14
                # print(f"self.head_pitch_state_rad: {self.head_pitch_state_rad}")
                # print(f"self.head_yaw_state_rad: {self.head_yaw_state_rad}")
                # print(f"tgt_head_pitch_rad------action: {tgt_head_pitch_rad}")
                # print(f"tgt_head_yaw_rad------action: {tgt_head_yaw_rad}")

                if self.emergency_stop:
                    if self.env.simulated:
                        self.env.set_root_fixed(True)
                    print('Emergency stopped')
                    self.env.step(joint_data['joint_pos'], 0, 0)
                    # torch.save(all_actions, "all_actions.pt")
                    # torch.save(all_joint_pos, "all_joint_pos.pt")
                    # torch.save(all_joint_vel, "all_joint_vel.pt")
                    break

                if self.env.simulated:
                    for i, sphere in enumerate(self.debug_spheres):
                        sphere_pos = motion_rb_pos[0, i].cpu().numpy().copy()
                        sphere.pos = sphere_pos

                intermediates = self.intermediates.compute(joint_data['joint_pos'], 
                                                        joint_data['joint_vel'], 
                                                        root_data['root_quat'], 
                                                        root_data['root_ang_vel'], 
                                                        self.last_action)
                observation = compute_observation(**intermediates,
                                                motion_rb_pos=motion_rb_pos, 
                                                motion_rb_quat=motion_rb_quat,  
                                                root_lin_vel=motion_rb_lin_vel[:, 0])


                for k, v in observation.items():
                    observation[k] = v.to(self.device).unsqueeze(0)
                action = self.policy(**observation)[0].clamp(-10.0, 10.0)
                # all_actions.append(action.cpu().clone())
                # all_joint_pos.append(joint_data['joint_pos'].cpu().clone())
                # all_joint_vel.append(joint_data['joint_vel'].cpu().clone())

                self.last_action = action.cpu().clone()

                self.env.step(action, float(38 - tgt_head_pitch_deg), float(tgt_head_yaw_deg))

                if self.showing_image:
                    self.collector.show_image()
                
                if time.monotonic() - start_policy_time > 1.0:
                    print(f"Policy Frequency: {self.env.step_frequency}, Episode Step Frequency: {episode_freq}")
                    start_policy_time = time.monotonic()

            episode_end_timer = time.time()
            try:
                # print(f"\rEpisode Step Frequency: ", 1 / (episode_end_timer - episode_start_timer), end="")
                episode_freq = 1 / (episode_end_timer - episode_start_timer)    
            except Exception as e:
                print(e)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pick up a pink toy rabbit and put to the green bowl", help="Name of the task.")
    parser.add_argument("--checkpoint", type=str, help="Pytorch model checkpoint to load.")
    parser.add_argument("--vlm_server_ip", type=str, default="10.1.101.144")
    parser.add_argument("--vlm_server_port", type=int, default=6666)
    parser.add_argument("--policy_path", type=str, default="models/policy_twist&our.pt")
    parser.add_argument("--episode_length", type=int, default=40000, help="Step horizon of each rollout.")
    parser.add_argument("--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor.")
    parser.add_argument("--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for inference.")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode.")
    parser.add_argument("--replay", action="store_true", default=False, help="Replay mode.")
    parser.add_argument("--replay_state", action="store_true", default=False, help="Replay state mode.")
    parser.add_argument("--record", action="store_true", default=False, help="Record mode.")
    parser.add_argument("--action_horizon", type=int, default=8, help="Action horizon.")
    parser.add_argument("--robot_type", type=str, default="unitree_g1", help="Robot type.")

    args_cli = parser.parse_args()

    server = G1_Vla_Server(args=args_cli, device=args_cli.device)

    try:
        server.run_main_loop()
    finally:
        server.hand_service.terminate()
        server.hand_qpos_shm.close()
        server.hand_qpos_shm.unlink()
        server.vla_client.kill_server()