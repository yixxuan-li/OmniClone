#!/usr/bin/env python3
"""
Video Recorder Script
Receives timestamped video stream from the C++ sender and records it.
Protocol: [8B timestamp BE][4B len BE][payload]
"""
import argparse
import socket
import struct
import cv2
import numpy as np
import time
import subprocess
import threading
import queue
import os
import fcntl
from datetime import datetime
from pathlib import Path
from multiprocessing import shared_memory
import select


MAX_LEN = 256
PATH_HEADER_SIZE = 8  # 4 bytes for length + 4 bytes for ready flag
class ImageRecorder:
    def __init__(
        self,
        host='',
        port=12345,
        output_dir='recordings',
        save_frames=False,
        display=False,
        codec='H264',
        image_format='jpg',
        max_display_fps=30,
        skip_frames=True,
        width=640,
        height=480,
        img_shm_name=None,
        timestamp_shm_name=None,
        img_path_shm_name=None,
        frame_seq_shm_name=None,
    ):
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.save_frames = save_frames
        self.display = display
        self.codec = codec.upper()
        self.image_format = image_format
        self.skip_frames = skip_frames

        self.frame_width = width
        self.frame_height = height
        self.raw_frame_size = width * height * 3

        self.connected = False
        self.sock = None
        self.server_sock = None

        self.frame_count = 0
        self.img_shown_count = 0
        self.start_time = None
        self.last_display_time = 0
        self.display_interval = 1.0 / max_display_fps if max_display_fps > 0 else 0

        # queues
        self._encoded_queue = queue.Queue(maxsize=100)
        self._decoded_queue = queue.Queue(maxsize=100)
        self._ts_fifo = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()

        self._raw_buf = bytearray()

        self.stats = {
            "frames_received": 0,
            "frames_skipped": 0,
            "decode_errors": 0,
        }

        self.ffmpeg_proc = None
        self._start_ffmpeg()

        # self._decode_thread = threading.Thread(
        #     target=self._ffmpeg_decode_loop, daemon=True
        # )
        # self._decode_thread.start()

        # shared memory
        self.color_array = None
        if img_shm_name:
            self.shm_img = shared_memory.SharedMemory(name=img_shm_name)
            self.color_array = np.ndarray(
                (height, width, 3), dtype=np.uint8, buffer=self.shm_img.buf
            )

        self.timestamp_array = None
        if timestamp_shm_name:
            self.shm_ts = shared_memory.SharedMemory(name=timestamp_shm_name)
            self.timestamp_array = np.ndarray((1,), dtype=np.float64, buffer=self.shm_ts.buf)

        self.img_path = None
        self.img_path_buf = None
        if img_path_shm_name:
            self.img_path_shm = shared_memory.SharedMemory(name=img_path_shm_name)
            self.img_path_buf = np.ndarray((MAX_LEN + PATH_HEADER_SIZE,), dtype=np.uint8, buffer=self.img_path_shm.buf)
        
        # Frame sequence number for synchronization
        self.frame_seq_array = None
        if frame_seq_shm_name:
            self.frame_seq_shm = shared_memory.SharedMemory(name=frame_seq_shm_name)
            self.frame_seq_array = np.ndarray((1,), dtype=np.uint64, buffer=self.frame_seq_shm.buf)

        self._ffmpeg_thread = threading.Thread(
            target=self._ffmpeg_loop, daemon=True
        )
        self._ffmpeg_thread.start()

        # Separate thread for processing decoded frames (avoids blocking on network recv)
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._process_thread.start()

    # image path reader with ready flag check
    def read_str_if_ready(self):
        """Read path string only if ready flag is set. Returns None if not ready."""
        if self.img_path_buf is not None:
            # Check ready flag
            ready_flag = struct.unpack_from("I", self.img_path_buf, 4)[0]
            if ready_flag == 0:
                return None  # Not ready
            
            # Read length and string
            length = struct.unpack_from("I", self.img_path_buf, 0)[0]
            path = bytes(self.img_path_buf[PATH_HEADER_SIZE:PATH_HEADER_SIZE+length]).decode()
            
            # Clear ready flag after reading (acknowledge receipt)
            struct.pack_into("I", self.img_path_buf, 4, 0)
            
            return path
        return None
    
    def read_str(self):
        """Legacy read method - reads without checking ready flag."""
        if self.img_path_buf is not None:
            length = struct.unpack_from("I", self.img_path_buf, 0)[0]
            return bytes(self.img_path_buf[PATH_HEADER_SIZE:PATH_HEADER_SIZE+length]).decode()
        return None
    # ------------------------------------------------------------
    # ffmpeg
    # ------------------------------------------------------------
    def _start_ffmpeg(self):
        codec_name = "h264" if self.codec == "H264" else "hevc"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",

            # Key 1: Disable ffmpeg GOP etc
            "-fflags", "+nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-strict", "experimental",

            # Key 2: Force decode by packet
            "-use_wallclock_as_timestamps", "1",

            "-f", codec_name,
            "-i", "pipe:0",

            # Key 3: Disable internal frame buffer
            "-vsync", "0",

            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]

        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        fd = self.ffmpeg_proc.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        
        # Also set stdin to non-blocking to prevent write() from blocking
        stdin_fd = self.ffmpeg_proc.stdin.fileno()
        stdin_flags = fcntl.fcntl(stdin_fd, fcntl.F_GETFL)
        fcntl.fcntl(stdin_fd, fcntl.F_SETFL, stdin_flags | os.O_NONBLOCK)

        print("[Recorder] ffmpeg started")

    def _restart_ffmpeg(self):
        print("[Recorder] restarting ffmpeg")
        try:
            self.ffmpeg_proc.kill()
        except Exception:
            pass
        self._raw_buf.clear()
        self._start_ffmpeg()

    def _ffmpeg_loop(self):
        """
        Single-threaded loop:
        - take encoded packets from self._encoded_queue (non-blocking)
        - write to ffmpeg.stdin
        - wait on ffmpeg.stdout (select), read chunks, append to internal raw buffer
        - when raw buffer has a full frame, extract, make a safe copy, and:
            - write into shared memory (self.color_array) if present
            - update timestamp_array if present
            - push into decoded queue
        Replaces previous two-thread design to avoid concurrent access to self._raw_buf.
        """
        fd = self.ffmpeg_proc.stdout.fileno()
        frame_bytes_needed = self.raw_frame_size
        # Local alias for speed
        raw_buf = self._raw_buf
        
        # Initialize timestamp_ms to avoid scope issues
        timestamp_ms = 0
        
        while not self._stop_event.is_set():
            # 1) Try to quickly flush any pending encoded packets into ffmpeg.stdin
            # Use non-blocking get to avoid blocking here

            try:
                while True:
                    timestamp_ms, encoded = self._encoded_queue.get_nowait()
                    try:
                        self.ffmpeg_proc.stdin.write(b"\x00\x00\x00\x01" + encoded)
                        self.ffmpeg_proc.stdin.flush()
                    except BlockingIOError:
                        # stdin buffer full, put data back and try later
                        try:
                            self._encoded_queue.put_nowait((timestamp_ms, encoded))
                        except queue.Full:
                            pass  # drop if queue full
                        break
                    except Exception:
                        # If ffmpeg broken, restart and stop draining more encoded packets now
                        self._restart_ffmpeg()
                        # refresh fd and continue outer loop
                        fd = self.ffmpeg_proc.stdout.fileno()
                        break
            except queue.Empty:
                pass

            # 2) Wait for ffmpeg stdout to be readable (small timeout to periodically re-check encoded queue)
            try:
                rlist, _, _ = select.select([fd], [], [], 0.02)
            except ValueError:
                # ffmpeg process might have exited; try restart
                if not self._stop_event.is_set():
                    self._restart_ffmpeg()
                    fd = self.ffmpeg_proc.stdout.fileno()
                continue
            # print(f"[Debug] rlist: {rlist}")
            if rlist:
                # read available chunk(s)
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    chunk = b""
                except OSError:
                    # ffmpeg stdout likely closed; restart and continue
                    if not self._stop_event.is_set():
                        self._restart_ffmpeg()
                        fd = self.ffmpeg_proc.stdout.fileno()
                    continue

                if not chunk:
                    # nothing read (EOF or no data) — small sleep and continue
                    time.sleep(0.001)
                    continue

                # append to raw buffer (single-threaded, safe)
                raw_buf.extend(chunk)

                # 3) while a full raw frame exists, extract and process it
                while len(raw_buf) >= frame_bytes_needed and not self._stop_event.is_set():
                    # Make a bytes copy of the frame bytes to ensure the memory is owned and stable
                    # (this avoids creating a numpy view pointing into raw_buf which might be mutated)
                    frame_bytes = bytes(raw_buf[:frame_bytes_needed])
                    # remove consumed bytes from raw_buf
                    del raw_buf[:frame_bytes_needed]

                    # create numpy array from bytes and copy into a new owned array (copy() ensures OOM-safe)
                    try:
                        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                            self.frame_height, self.frame_width, 3
                        )
                        # ensure owned buffer
                        frame = frame.copy()
                    except Exception as e:
                        # decoding/reshape error — skip this frame
                        self.stats["decode_errors"] += 1
                        continue


                    # also put into decoded queue for local display/processing if needed
                    if self._decoded_queue is not None:
                        if self._decoded_queue.full():
                            try:
                                self._decoded_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self._decoded_queue.put((timestamp_ms, frame))

            else:
                # select timed out — small sleep prevents busy loop
                time.sleep(0.001)
        # loop exit: clean up
        try:
            if self.ffmpeg_proc:
                try:
                    self.ffmpeg_proc.stdin.close()
                except Exception:
                    pass
                try:
                    self.ffmpeg_proc.kill()
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------
    # networking
    # ------------------------------------------------------------
    def listen(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)

        print(f"Listening on {self.host or '0.0.0.0'}:{self.port}")
        self.sock, addr = self.server_sock.accept()
        self.connected = True
        print("Sender connected:", addr)
        return True

    def receive_and_process(self):
        buf = b""
        HEADER = 12

        try:
            while self.connected:
                data = self.sock.recv(65536)
                if not data:
                    break
                buf += data
                while len(buf) >= HEADER:
                    ts = struct.unpack(">Q", buf[:8])[0]
                    size = struct.unpack(">I", buf[8:12])[0]

                    if len(buf) < HEADER + size:
                        break

                    payload = buf[12 : 12 + size]
                    buf = buf[12 + size :]
                    # print(f"[RX] encoded size={len(payload)} ts={ts}")
                    try:
                        self._encoded_queue.put((ts, payload), timeout=0.01)
                    except queue.Full:
                        self.stats["frames_skipped"] += 1

                # process decoded
                while not self._decoded_queue.empty():
                    ts, frame = self._decoded_queue.get_nowait()
                    self._process_frame(ts, frame)

        finally:
            self._stop_event.set()
            self._ffmpeg_thread.join(timeout=1)
            self.disconnect()

    # ------------------------------------------------------------
    # frame handling
    # ------------------------------------------------------------
    def _process_loop(self):
        """Separate thread to process decoded frames from the queue.
        This ensures frame processing is not blocked by network recv.
        """
        while not self._stop_event.is_set():
            try:
                ts, frame = self._decoded_queue.get(timeout=0.05)
                self._process_frame(ts, frame)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Recorder] Error in process loop: {e}")
                continue
    
    def _process_frame(self, timestamp_ms, frame):
        # print(f"[Debug] _process_frame called, ts={timestamp_ms}")
        if self.frame_count == 0:
            readable = datetime.fromtimestamp(timestamp_ms / 1000.0)
            print(f"First frame @ {readable}")

        # Write frame to shared memory
        if self.color_array is not None:
            np.copyto(self.color_array, frame)

        # Update timestamp
        if self.timestamp_array is not None:
            self.timestamp_array[0] = timestamp_ms
        
        # Increment frame sequence number AFTER writing frame and timestamp
        # This signals to the reader that a complete new frame is available
        if self.frame_seq_array is not None:
            self.frame_seq_array[0] += 1
        
        # Check if main process has requested us to save an image
        # Use read_str_if_ready to avoid race condition
        new_path = self.read_str_if_ready()
        if new_path is not None:
            self.img_path = new_path
        
        # Save frame to disk if path is set and not "not_start"
        if frame is None:
            print('frame is None')
        elif self.img_path is not None and self.img_path != "not_start":
            try:
                cv2.imwrite(self.img_path, frame)
            except Exception as e:
                print(f"[Error] Failed to write image: {e}")


        if self.display:
            now = time.time()
            if now - self.last_display_time >= self.display_interval:
                cv2.imshow("Video", frame)
                self.last_display_time = now
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.connected = False

        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 1
            fps = self.frame_count / elapsed
            # print(
            #     f"Frame {self.frame_count} ts={timestamp_ms} fps={fps:.2f} "
            #     f"q_enc={self._encoded_queue.qsize()} q_dec={self._decoded_queue.qsize()}"
            # )

        if self.start_time is None:
            self.start_time = time.time()

        self.frame_count += 1

    # ------------------------------------------------------------
    def disconnect(self):
        self.connected = False
        self._stop_event.set()  # Signal threads to stop
        
        # Wait for ffmpeg thread to finish
        try:
            if hasattr(self, '_ffmpeg_thread') and self._ffmpeg_thread.is_alive():
                self._ffmpeg_thread.join(timeout=1.0)
        except Exception:
            pass
        
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self.server_sock.close()
        except Exception:
            pass
        try:
            self.ffmpeg_proc.kill()
        except Exception:
            pass
        
        # Close SharedMemory references (don't unlink - main process owns them)
        try:
            if hasattr(self, 'shm_img'):
                self.shm_img.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'shm_ts'):
                self.shm_ts.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'img_path_shm'):
                self.img_path_shm.close()
        except Exception:
            pass
        try:
            if hasattr(self, 'frame_seq_shm'):
                self.frame_seq_shm.close()
        except Exception:
            pass
        
        print("Disconnected")



def main():
    parser = argparse.ArgumentParser(description='Video Recorder - Receives timestamped video stream')
    parser.add_argument('--host', type=str, default='',
                        help='IP address to bind to (default: empty = listen on all interfaces)')
    parser.add_argument('--port', type=int, default=12345,
                        help='Port to listen on (default: 12345)')
    parser.add_argument('--output', type=str, default='recordings',
                        help='Output directory for recordings (default: recordings)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Save frames to files (default: True)')
    parser.add_argument('--no-save', dest='save', action='store_false',
                        help='Do not save frames')
    parser.add_argument('--display', action='store_true',
                        help='Display frames (requires decoded frames)')
    parser.add_argument('--codec', type=str, default='H264', choices=['H264', 'H265', 'HEVC'],
                        help='Video codec (default: H264)')
    parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'jpeg', 'png'],
                        help='Image format for saved frames (default: jpg)')
    parser.add_argument('--max-fps', type=int, default=30,
                        help='Maximum display frame rate (default: 30)')
    parser.add_argument('--skip-frames', action='store_true', default=True,
                        help='Skip frames if decoding/display can\'t keep up')
    parser.add_argument('--width', type=int, default=640,
                        help='Video frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Video frame height (default: 480)')
    args = parser.parse_args()

    img_shape = (480, 640, 3)
    img_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)

    timestamp_shm = shared_memory.SharedMemory(create=True, size=np.float64().itemsize)
    timestamp_array = np.ndarray((1,), dtype=np.float64, buffer=timestamp_shm.buf)

    recorder = ImageRecorder(
        host=args.host,
        port=args.port,
        output_dir=args.output,
        save_frames=args.save,
        display=args.display,
        codec=args.codec,
        image_format=args.format,
        max_display_fps=args.max_fps,
        skip_frames=args.skip_frames,
        width=args.width,
        height=args.height,
        img_shm_name=img_shm.name,
        timestamp_shm_name=timestamp_shm.name
    )
    
    if recorder.listen():
        recorder.receive_and_process()
    else:
        print("Failed to listen/accept connection. Exiting.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

