from .episode_writer import EpisodeWriter
from .image_recorder import ImageRecorder
from multiprocessing import Process, shared_memory, Array

import numpy as np
import time
import cv2
import struct
import signal

def image_recorder_process_main(recorder_cfg):
    recorder = None

    def cleanup(signum=None, frame=None):
        """Handle termination signal and cleanup resources."""
        if recorder is not None:
            try:
                recorder.disconnect()
            except Exception:
                pass
        if signum is not None:
            exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        recorder = ImageRecorder(**recorder_cfg)
        if not recorder.listen():
            print("listening not ok")
            return
        recorder.receive_and_process()
    except Exception as e:
        print(f"[ImageRecorder] Error: {e}")
    finally:
        cleanup()

def is_pure_black(img: np.ndarray) -> bool:
    img = np.asarray(img)
    if img.size == 0:
        return True
    return int(img.max()) == 0

MAX_LEN = 256
PATH_HEADER_SIZE = 8  # 4 bytes for length + 4 bytes for ready flag



class VlaDataCollector:
    def __init__(self, data_dir: str, task_desc: str, frequency: int, p_gains: list, d_gains: list):
        self.episode_writer = EpisodeWriter(task_dir=data_dir, frequency=frequency)
        self.episode_writer.set_task_description(task_desc)
        self.episode_writer.set_pd_gains(p_gains, d_gains)
        self.is_collecting = False
        self.image_count = 0
        self.frame_loss_count = 0
        self.is_initialized = False  # Will be set to True if camera validation passes
        self.start_realsense_process()

    def __del__(self):
        """Ensure SharedMemory resources are properly cleaned up."""
        try:
            self.stop_realsense_process()
        except Exception:
            pass

    def start_realsense_process(self):
        img_shape = (480, 640, 3)

        # shared image buffer
        self.img_shm = shared_memory.SharedMemory(
            create=True,
            size=np.prod(img_shape) * np.uint8().itemsize
        )
        self.img_array = np.ndarray(
            img_shape,
            dtype=np.uint8,
            buffer=self.img_shm.buf
        )
        self.img_array[:] = 0

        # timestamp buffer
        self.timestamp_shm = shared_memory.SharedMemory(
            create=True,
            size=np.float64().itemsize
        )
        self.timestamp_array = np.ndarray(
            (1,),
            dtype=np.float64,
            buffer=self.timestamp_shm.buf
        )
        self.timestamp_array[0] = 0.0
        
        # Frame sequence number for detecting new frames (avoids race condition)
        self.frame_seq_shm = shared_memory.SharedMemory(create=True, size=8)  # uint64
        self.frame_seq_array = np.ndarray((1,), dtype=np.uint64, buffer=self.frame_seq_shm.buf)
        self.frame_seq_array[0] = 0
        
        # Image path buffer with ready flag
        # Structure: [4B length][4B ready_flag][path_string...]
        self.img_path_shm = shared_memory.SharedMemory(create=True, size=MAX_LEN + PATH_HEADER_SIZE)
        self.img_path_buf = np.ndarray((MAX_LEN + PATH_HEADER_SIZE,), dtype=np.uint8, buffer=self.img_path_shm.buf)
        self.img_path_buf[:] = 0
        self._write_path_with_flag("not_start", ready=False)

        recorder_cfg = dict(
            host='',
            port=12345,
            output_dir='vla_data',
            save_frames=True,
            display=False,
            codec='H264',
            image_format='jpg',
            max_display_fps=30,
            skip_frames=False,
            width=640,
            height=480,
            img_shm_name=self.img_shm.name,
            timestamp_shm_name=self.timestamp_shm.name,
            img_path_shm_name=self.img_path_shm.name,
            frame_seq_shm_name=self.frame_seq_shm.name
        )

        self.realsense_process = Process(
            target=image_recorder_process_main,
            args=(recorder_cfg,),
            daemon=True
        )
        self.realsense_process.start()
        
        # Validate camera is working with retries
        max_retries = 5
        retry_delay = 10.0  # seconds
        
        for attempt in range(max_retries):
            time.sleep(retry_delay)
            
            # Check if process is alive
            if not self.realsense_process.is_alive():
                print(f"[Data Collector] Process not alive, restarting (attempt {attempt + 1}/{max_retries})")
                self._restart_realsense_process(recorder_cfg)
                continue
            
            # Check if timestamp is being updated (indicates data is flowing)
            if self.timestamp_array[0] == 0.0:
                print(f"[Data Collector] Waiting for camera data (attempt {attempt + 1}/{max_retries})")
                continue
            
            # Check if image is valid (not pure black)
            if is_pure_black(self.img_array):
                print(f"[Data Collector] Image is black, restarting (attempt {attempt + 1}/{max_retries})")
                self._restart_realsense_process(recorder_cfg)
                continue
            
            # All checks passed
            print("[Data Collector] Camera validated successfully")
            self.is_initialized = True
            return
        
        # All retries exhausted
        self.is_initialized = False
        print("[Error] Failed to validate camera after retries")
    
    def _restart_realsense_process(self, recorder_cfg):
        """Helper to restart the realsense process during validation."""
        if self.realsense_process is not None:
            self.realsense_process.terminate()
            self.realsense_process.join(timeout=1.0)
        
        self.img_array[:] = 0
        self.timestamp_array[0] = 0.0
        
        self.realsense_process = Process(
            target=image_recorder_process_main,
            args=(recorder_cfg,),
            daemon=True
        )
        self.realsense_process.start()
    
    def _write_path_with_flag(self, s: str, ready: bool = True):
        """Write path string with ready flag for synchronization."""
        data = s.encode("utf-8")
        n = len(data)
        if n + PATH_HEADER_SIZE > MAX_LEN + PATH_HEADER_SIZE:
            raise ValueError(f"String too long: {n} bytes (max {MAX_LEN})")
        # Write length
        struct.pack_into("I", self.img_path_buf, 0, n)
        # Write ready flag (0 = not ready, 1 = ready)
        struct.pack_into("I", self.img_path_buf, 4, 1 if ready else 0)
        # Write string data
        self.img_path_buf[PATH_HEADER_SIZE:PATH_HEADER_SIZE+n] = np.frombuffer(data, np.uint8)
    
    def write_str(self, s: str):
        """Write path and mark as ready for ImageRecorder to read."""
        self._write_path_with_flag(s, ready=True)

    def start_collection(self, state_msg_tick):
        self.episode_writer.create_episode()
        self.is_collecting = True
        self.image_count = 0
        self.frame_loss_count = 0
        self.camera_start_time = self.timestamp_array[0]
        print(f"[Data Collector] Camera start time: {self.camera_start_time}")
        self.state_start_tick = state_msg_tick
    
    def stop_realsense_process(self):
        if self.realsense_process is not None:
            self.realsense_process.terminate()
            self.realsense_process.join(timeout=1.0)
            self.realsense_process = None

        if hasattr(self, "img_shm"):
            self.img_shm.close()
            self.img_shm.unlink()

        if hasattr(self, "timestamp_shm"):
            self.timestamp_shm.close()
            self.timestamp_shm.unlink()
        
        if hasattr(self, "img_path_shm"):
            self.img_path_shm.close()
            self.img_path_shm.unlink()
        
        if hasattr(self, "frame_seq_shm"):
            self.frame_seq_shm.close()
            self.frame_seq_shm.unlink()

    # task_obs is customized
    def add_data_to_episode(self, states, actions, task_obs, state_msg_tick):
        # Track frame sequence to know if we're reading a fresh frame
        # Note: Images are ~30Hz, states are 50Hz, so some frames may be reused
        # The frame_seq ensures we only read AFTER the writer has finished writing
        # current_frame_seq = self.frame_seq_array[0]
        # if not hasattr(self, '_last_frame_seq'):
        #     self._last_frame_seq = 0
        
        # is_new_frame = (current_frame_seq != self._last_frame_seq)
        # if not is_new_frame:
        #     self.frame_loss_count += 1
        # else:
        #     self.frame_loss_count = 0  # Reset count when new frame received, only detect continuous frame loss
        # if self.frame_loss_count > 150:
        #     print("[Data Collector] Frame loss count: {}".format(self.frame_loss_count))
        #     return False
        
        # self._last_frame_seq = current_frame_seq
        
        # Always proceed - even if no new image frame, we still save states at 50Hz
        # The frame_seq check just ensures we're reading complete (not partial) data
        # current_tv_image = self.img_array
        self.image_count += 1
        current_timestamp_image = self.timestamp_array[0] - self.camera_start_time
        state_reference_tick = state_msg_tick - self.state_start_tick
        colors = {}
        depths = {}
        colors[f"color_{0}"] = "current_tv_image"
        
        current_img_path = self.episode_writer.add_item(colors=colors, depths=depths, states=states, actions=actions, \
            task_obs=task_obs, timestamp_img=current_timestamp_image, \
            timestamp_state=state_reference_tick, timestamp_servo=state_reference_tick)
        
        self.write_str(current_img_path)
        
        return True
    
    # def show_image(self):
    #     self.image_recorder._show_image()
        
    
    def save_episode(self):
        self.write_str("not_start")
        self.episode_writer.save_episode()
        self.is_collecting = False
        self.state_start_tick = 0
        self.image_count = 0
    