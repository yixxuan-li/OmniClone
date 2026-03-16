from dex_retargeting import RetargetingConfig
from pathlib import Path
import yaml
from enum import Enum
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

class HandType(Enum):
    INSPIRE_HAND = "./assets/inspire_hand/inspire_hand.yml"
    INSPIRE_HAND_Unit_Test = "../../assets/inspire_hand/inspire_hand.yml"
    UNITREE_DEX3 = "./assets/unitree_hand/unitree_dex3.yml"
    UNITREE_DEX3_Unit_Test = "../../assets/unitree_hand/unitree_dex3.yml"
    BRAINCO_HAND = "./assets/brainco_hand/brainco.yml"
    BRAINCO_HAND_Unit_Test = "../../assets/brainco_hand/brainco.yml"

class HandRetargeting:
    def __init__(self, hand_type: HandType):
        left_joint_order = None
        right_joint_order = None
        if hand_type == HandType.UNITREE_DEX3:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.UNITREE_DEX3_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('../../assets')
        elif hand_type == HandType.INSPIRE_HAND:
            RetargetingConfig.set_default_urdf_dir('./assets')
            left_joint_order = [
                "L_thumb_proximal_yaw_joint",   # 22
                "L_thumb_proximal_pitch_joint",   # 23
                "L_thumb_intermediate_joint",   # 24
                "L_index_proximal_joint",   # 25
                "L_index_intermediate_joint",   # 26
                "L_middle_proximal_joint",  # 27
                "L_middle_intermediate_joint",  # 28
                "L_ring_proximal_joint",    # 29
                "L_ring_intermediate_joint",    # 30
                "L_pinky_proximal_joint",  # 31
                "L_pinky_intermediate_joint",  # 32
            ]
            right_joint_order = [
                "R_thumb_proximal_yaw_joint",   # 40
                "R_thumb_proximal_pitch_joint",   # 41
                "R_thumb_intermediate_joint",   # 42
                "R_index_proximal_joint",   # 43
                "R_index_intermediate_joint",   # 44
                "R_middle_proximal_joint",  # 45
                "R_middle_intermediate_joint",  # 46
                "R_ring_proximal_joint",    # 47
                "R_ring_intermediate_joint",    # 48
                "R_pinky_proximal_joint",  # 49
                "R_pinky_intermediate_joint",  # 50
            ]

        elif hand_type == HandType.INSPIRE_HAND_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('../../assets')
        elif hand_type == HandType.BRAINCO_HAND:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.BRAINCO_HAND_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('../../assets')

        config_file_path = Path(hand_type.value)
        assert left_joint_order is not None and right_joint_order is not None, "Joint order is not set"

        try:
            with config_file_path.open('r') as f:
                self.cfg = yaml.safe_load(f)
                
            if 'left' not in self.cfg or 'right' not in self.cfg:
                raise ValueError("Configuration file must contain 'left' and 'right' keys.")

            left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
            right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()

            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            self.right_retargeting_joint_names = self.right_retargeting.joint_names
            self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
            self.right_indices = self.right_retargeting.optimizer.target_link_human_indices
            self.left_dex_retargeting_to_asset = [ self.left_retargeting_joint_names.index(name) for name in left_joint_order]
            self.right_dex_retargeting_to_asset = [ self.right_retargeting_joint_names.index(name) for name in right_joint_order]

            if hand_type == HandType.UNITREE_DEX3 or hand_type == HandType.UNITREE_DEX3_Unit_Test:
                # In section "Sort by message structure" of https://support.unitree.com/home/en/G1_developer/dexterous_hand
                self.left_dex3_api_joint_names  = [ 'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
                                                    'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 
                                                    'left_hand_index_0_joint', 'left_hand_index_1_joint' ]
                self.right_dex3_api_joint_names = [ 'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                                                    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
                                                    'right_hand_index_0_joint', 'right_hand_index_1_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_dex3_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_dex3_api_joint_names]

            elif hand_type == HandType.INSPIRE_HAND or hand_type == HandType.INSPIRE_HAND_Unit_Test:
                # "Joint Motor Sequence" of https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
                self.left_inspire_api_joint_names  = [ 'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint',
                                                       'L_index_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_proximal_yaw_joint' ]
                self.right_inspire_api_joint_names = [ 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint',
                                                       'R_index_proximal_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_proximal_yaw_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names]
            
            elif hand_type == HandType.BRAINCO_HAND or hand_type == HandType.BRAINCO_HAND_Unit_Test:
                # "Driver Motor ID" of https://www.brainco-hz.com/docs/revolimb-hand/product/parameters.html
                self.left_brainco_api_joint_names  = [ 'left_thumb_metacarpal_joint', 'left_thumb_proximal_joint', 'left_index_proximal_joint',
                                                       'left_middle_proximal_joint', 'left_ring_proximal_joint', 'left_pinky_proximal_joint' ]
                self.right_brainco_api_joint_names = [ 'right_thumb_metacarpal_joint', 'right_thumb_proximal_joint', 'right_index_proximal_joint',
                                                       'right_middle_proximal_joint', 'right_ring_proximal_joint', 'right_pinky_proximal_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_brainco_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_brainco_api_joint_names]
        
        except FileNotFoundError:
            logger_mp.warning(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            logger_mp.warning(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            logger_mp.error(f"An error occurred: {e}")
            raise

def normalize(val, min_val, max_val):
    return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

def normalize_inspire_hand(qpos):
    for idx in range(6):
        if idx <= 3:
            qpos[idx]  = normalize(qpos[idx], 0.0, 1.7)
        elif idx == 4:
            qpos[idx]  = normalize(qpos[idx], 0.0, 0.5)
        elif idx == 5:
            qpos[idx]  = normalize(qpos[idx], -0.1, 1.3)
    return qpos

def normalize_hand(qpos, hand_type: HandType):
    if hand_type == HandType.INSPIRE_HAND:
        return normalize_inspire_hand(qpos)
    elif hand_type == HandType.UNITREE_DEX3:
        return qpos
    elif hand_type == HandType.BRAINCO_HAND:
        return qpos
    else:
        raise ValueError(f"Invalid hand type: {hand_type}")

HAND_TYPE_MAP = {
    "inspire_hand": HandType.INSPIRE_HAND,
    "unitree_dex3": HandType.UNITREE_DEX3,
    "brainco_hand": HandType.BRAINCO_HAND,
}

# num_retarget_joints, num_asset_joints, num_real_joints
HAND_RETARGET_DIMS = {
    "inspire_hand": [12, 11, 6]
}
import numpy as np
from multiprocessing.shared_memory import SharedMemory
def start_retargeting(hand_type: str, hand_array_shm_name, hand_qpos_shm_name):
    num_retarget_joints, num_asset_joints, num_real_joints = HAND_RETARGET_DIMS[hand_type]
    hand_type = HAND_TYPE_MAP[hand_type]
    hand_retargeting = HandRetargeting(hand_type)

    assert num_retarget_joints == len(hand_retargeting.left_retargeting_joint_names), f'Expected {num_retarget_joints} retargeting joints, but got {len(hand_retargeting.left_retargeting_joint_names)}'
    assert num_asset_joints == len(hand_retargeting.left_dex_retargeting_to_asset), f'Expected {num_asset_joints} asset joints, but got {len(hand_retargeting.left_retargeting_joint_names)}'
    assert num_real_joints == len(hand_retargeting.left_dex_retargeting_to_hardware), f'Expected {num_real_joints} real joints, but got {len(hand_retargeting.left_retargeting_joint_names)}'

    hand_array_shm = SharedMemory(hand_array_shm_name)
    hand_qpos_shm = SharedMemory(hand_qpos_shm_name)
    hand_array = np.ndarray((2, 25, 3), dtype=np.float32, buffer=hand_array_shm.buf)
    hand_qpos = np.ndarray(2*(num_retarget_joints+num_asset_joints+num_real_joints), dtype=np.float32, buffer=hand_qpos_shm.buf)
    
    while True:
        left_hand_data = hand_array[0, hand_retargeting.left_indices[1,:]] - hand_array[0, hand_retargeting.left_indices[0,:]]
        right_hand_data = hand_array[1, hand_retargeting.right_indices[1,:]] - hand_array[1, hand_retargeting.right_indices[0,:]]
        left_qpos = hand_retargeting.left_retargeting.retarget(left_hand_data)
        right_qpos = hand_retargeting.right_retargeting.retarget(right_hand_data)
        
        hand_qpos[0:num_retarget_joints] = left_qpos
        hand_qpos[num_retarget_joints:num_retarget_joints+num_asset_joints] = left_qpos[hand_retargeting.left_dex_retargeting_to_asset]
        hand_qpos[num_retarget_joints+num_asset_joints:num_retarget_joints+num_asset_joints+num_real_joints] = normalize_hand(left_qpos[hand_retargeting.left_dex_retargeting_to_hardware], hand_type)

        right_start_index = num_retarget_joints+num_asset_joints+num_real_joints
        hand_qpos[right_start_index:right_start_index+num_retarget_joints] = right_qpos
        hand_qpos[right_start_index+num_retarget_joints:right_start_index+num_retarget_joints+num_asset_joints] = right_qpos[hand_retargeting.right_dex_retargeting_to_asset]
        hand_qpos[right_start_index+num_retarget_joints+num_asset_joints:right_start_index+num_retarget_joints+num_asset_joints+num_real_joints] = normalize_hand(right_qpos[hand_retargeting.right_dex_retargeting_to_hardware], hand_type)