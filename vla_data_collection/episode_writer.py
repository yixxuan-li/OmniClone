import os
import json
import datetime
import numpy as np
import torch

import multiprocessing
from multiprocessing import Value, Lock, Event

STOP = None

class EpisodeWriter(object):
    def __init__(self, task_dir, frequency=50, image_size=[640, 480], flush_every=10, queue_size=50):
        """
        image_size: [width, height]
        """
        print("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_size = image_size
        self.episode_live = False
        self.episode_event = Event()
        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        self.flush_every = flush_every
        self.queue_size = Value('i', 0)
        self.queue_lock = Lock()
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            print(f"==> task_dir directory already exist, now self.episode_id is: {self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            print(f"==> episode directory does not exist, now create one.\n")
        self.data_info()
        self.text_desc("")
        # self.save_queue = multiprocessing.Queue(maxsize=queue_size)
        self.item_queue = multiprocessing.Queue(maxsize=queue_size)
        # self._stop_event = threading.Event()
        self._writer_process = multiprocessing.Process(
            target=self._stream_loop,
            daemon=True,
        )
        self._writer_process.start()

        self.img_path = None
        print("==> EpisodeWriter initialized successfully.\n")

    def set_pd_gains(self, p_gains, d_gains):
        self.info["pd_params"]["p_gains"] = p_gains
        self.info["pd_params"]["d_gains"] = d_gains
    
    def set_task_description(self, task_desc):
        self.info["task_desc"] = task_desc
        self.text_desc(task_desc)
    
    def data_info(self, version='1.0.0', date=None, author="BIGAI"):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": "" if date is None else datetime.date.today().strftime('%Y-%m-%d'), 
                "author": "BIGAI" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   ['kLeftShoulderPitch' ,'kLeftShoulderRoll', 'kLeftShoulderYaw', 'kLeftElbow', 'kLeftWristRoll', 'kLeftWristPitch', 'kLeftWristyaw'],
                    "left_hand":  [],
                    "right_arm":  ['kRightShoulderPitch' ,'kRightShoulderRoll', 'kRightShoulderYaw', 'kRightElbow', 'kRightWristRoll','kRightWristPitch', 'kRightWristyaw'],
                    "right_hand": [],
                    "waist":      ['kWaistYaw', 'kWaistRoll', 'kWaistPitch'],
                    "left_leg":   ['kLeftHipPitch', 'kLeftHipRoll', 'kLeftHipYaw', 'kLeftKnee', 'kLeftAnklePitch', 'kLeftAnkleRoll'],
                    "right_leg":  ['kRightHipPitch', 'kRightHipRoll', 'kRightHipYaw', 'kRightKnee', 'kRightAnklePitch', 'kRightAnkleRoll'],
                    "head_servo": ['kHeadPitch', 'kHeadYaw'],
                    "imu":        ['kBodyAngVelRoll', 'kBodyAngVelPitch', 'kBodyAngVelPitch', 'kBodyRoll', 'kBodyPitch'],
                },
                "tactile_names": {
                    "left_hand": [],
                    "right_hand": [],
                }, 
                "task_obs": {
                    "head": ['kHeadX', 'kHeadY', 'kHeadZ'],
                    "left_hand": ['kLeftHandX', 'kLeftHandY', 'kLeftHandZ'],
                    "right_hand": ['kRightHandX', 'kRightHandY', 'kRightHandZ'],
                },
                "pd_params": {
                    "p_gains": [],
                    "d_gains": [],
                },
                "task_desc": ""
            }
    def text_desc(self, task_desc):
        self.text = {
            "goal": task_desc,
            "desc": task_desc,
            "steps":task_desc,
        }

 
    def create_episode(self):
        """
        Create a new episode, each episode needs to specify the episode_id.
            text: Text descriptions of operation goals, steps, etc. The text description of each episode is the same.
            goal: operation goal
            desc: description
            steps: operation steps
        """
        self.episode_live = True
        self.episode_event.set()
        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
    
    def tensor_to_json(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [self.tensor_to_json(i) for i in x]
        if isinstance(x, dict):
            return {k: self.tensor_to_json(v) for k, v in x.items()}
        return x
    def fast_tensor_to_json(self, x):
        """
        High-speed tensor -> Python native type conversion, for real-time collection pipeline.
        Supports:
        - torch.Tensor (GPU/CPU)
        - numpy.ndarray
        - dict / list / tuple nested
        """
        import torch
        import numpy as np

        if x is None:
            return None

        if isinstance(x, torch.Tensor):
            # Convert to CPU numpy at once, then to list, faster than x.tolist()
            return x.detach().cpu().numpy().tolist()

        if isinstance(x, np.ndarray):
            return x.tolist()

        if isinstance(x, (list, tuple)):
            # Use list comprehension instead of recursive function call
            return [self.fast_tensor_to_json(i) for i in x]

        if isinstance(x, dict):
            # dict comprehension
            return {k: self.fast_tensor_to_json(v) for k, v in x.items()}

        # Return other types as is
        return x

    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, task_obs=None, log = False, \
        timestamp_img=None, timestamp_state=None, timestamp_servo=None):
        self.item_id += 1
        item_data = {
            'json_path': self.json_path,
            'idx': self.item_id,
            'colors': {"color_0": None},
            'depths': {"depth_0": None},
            'audios': {"mic0": None},
            'states': self.tensor_to_json(states),
            'actions': self.tensor_to_json(actions),
            'tactiles': {'left_hand':[], 'right_hand':[]},
            'task_obs': self.tensor_to_json(task_obs),
            'timestamp_img': timestamp_img,
            'timestamp_state': timestamp_state,
            'timestamp_servo': timestamp_servo,
        }

        # save images
        if colors is not None:
            for idx, (_, color) in enumerate(colors.items()):
                color_key = f'color_{idx}'
                color_name = f'{str(self.item_id).zfill(6)}_{color_key}.jpg'
                self.img_path = os.path.join(self.color_dir, color_name)
                # cv2.imwrite(self.img_path, color)
                item_data['colors'][color_key] = os.path.join('colors', color_name)

        # # save depths
        # if depths is not None:
        #     for idx, (_, depth) in enumerate(depths.items()):
        #         depth_key = f'depth_{idx}'
        #         depth_name = f'{str(self.item_id).zfill(6)}_{depth_key}.png'
        #         cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth)
        #         item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        # # save audios
        # if audios is not None:
        #     for mic, audio in audios.items():
        #         audio_name = f'audio_{str(self.item_id).zfill(6)}_{mic}.npy'
        #         np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
        #         item_data['audios'][mic] = os.path.join('audios', audio_name)

        # item_data['states'] = self.tensor_to_json(states)
        # item_data['actions'] = self.tensor_to_json(actions)
        # item_data['task_obs'] = self.tensor_to_json(task_obs)
        # item_data['tactiles'] = tactiles

        # self.queue.put(
        #     (
        #         item_data,
        #         colors,
        #         depths,
        #         audios,
        #     )
        # )
        
        # self.episode_data.append(item_data)
        with self.queue_lock:
            self.item_queue.put(item_data)
            self.queue_size.value += 1
            # print(f'len of item queue: {self.queue_size.value}')

        if log:
            curent_record_time = time.time()
            print(f"==> episode_id:{self.episode_id}  item_id:{self.item_id}  current_time:{curent_record_time}")
        
        return self.img_path


    def _writer_loop(self):
        while True:
            current_data = self.save_queue.get()  # blocking for data
            if current_data is None:
                break
            json_path = self.json_path
            with open(json_path, 'w', encoding='utf-8') as jsonf:
                json.dump(current_data, jsonf, indent=4, ensure_ascii=False)

            print('Current episode data saved.')
    
    def _stream_loop(self):
        while True:
            self.episode_event.wait()
            
            # Wait for the first valid item (skip any leftover STOP signals from previous episode)
            first_item = None
            while first_item is None:
                first_item = self.item_queue.get()
                if first_item is None:
                    # This is a STOP signal from previous episode, skip it
                    continue
                with self.queue_lock:
                    self.queue_size.value -= 1
            
            json_path = first_item['json_path']
            print(f'record item data into {json_path}')
            with open(json_path, 'w', encoding='utf-8') as jsonf:
                jsonf.write("{\n")
                jsonf.write(f'"info": {json.dumps(self.info)},\n')
                jsonf.write(f'"text": {json.dumps(self.text)},\n')
                jsonf.write('"data": [\n')
                jsonf.write(f"{json.dumps(first_item)}")
                while self.episode_event.is_set():
                    current_item = self.item_queue.get()
                    if current_item is None:
                        break
                    with self.queue_lock:
                        self.queue_size.value -= 1
                    jsonf.write(",\n")
                    jsonf.write(f"{json.dumps(current_item)}")
                jsonf.write("\n]\n")
                jsonf.write("}\n")
                jsonf.close()
                print(f'Current episode data saved in {json_path}.')


            

    def save_episode(self):
        """
            with open("./hmm.json",'r',encoding='utf-8') as json_file:
                model=json.load(json_file)
        """
        print("Starting VLA data saving.")
        # episode_data = self.episode_data
        # self.episode_data = []
        # episode_snapshot = {
        #     "info": self.info,
        #     "text": self.text,
        #     # "json_path": self.json_path,
        #     "data": episode_data,
        # }
        # self.save_queue.put(episode_snapshot)
        # Don't save date to disk here, because the IO will block the main loop
        # with open(self.json_path,'w',encoding='utf-8') as jsonf:
        #     # find the illigal type for saving json file
        #     # result = find_type_paths(self.data, target_type_name="SynchronizedArray", max_depth=10)
        #     # print(result)
        #     jsonf.write(json.dumps(self.data, indent=4, ensure_ascii=False))
        print("VLA data saving to disk in subprogress.")
        self.episode_live = False
        self.episode_event.clear()
        self.item_queue.put(STOP)

def find_type_paths(obj, target_type_name="SynchronizedArray", max_depth=10):
    results = []
    seen = set()
    def _walk(o, path, depth):
        if depth > max_depth:
            return
        oid = id(o)
        if oid in seen:
            return
        seen.add(oid)

        tname = type(o).__name__
        if target_type_name in tname:
            results.append((path, tname, type(o).__module__))
            return

        # dict-like
        if isinstance(o, dict):
            for k, v in o.items():
                _walk(v, f"{path}[{repr(k)}]", depth+1)
            return

        # list/tuple/set
        if isinstance(o, (list, tuple, set)):
            for i, v in enumerate(o):
                _walk(v, f"{path}[{i}]", depth+1)
            return

        # namedtuple or other sequence-like (fallback)
        if isinstance(o, collections.abc.Sequence) and not isinstance(o, (str, bytes, bytearray)):
            try:
                for i, v in enumerate(o):
                    _walk(v, f"{path}[{i}]", depth+1)
                return
            except TypeError:
                pass

        # object with __dict__
        if hasattr(o, "__dict__"):
            for attr, val in vars(o).items():
                _walk(val, f"{path}.{attr}", depth+1)
            return

        # other: nothing to do

    _walk(obj, "root", 0)
    return results