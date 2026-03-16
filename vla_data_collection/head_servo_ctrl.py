import sys
import select
import rclpy
from rclpy.node import Node
from unitree_hg.msg import (
    LowState,
    MotorState,
    LowCmd,
    MotorCmd,
)
from unitree_go.msg import (
    MotorCmds,
    MotorStates,
)
from unitree_go.msg import MotorState as MotorState_go
from unitree_go.msg import MotorCmd as MotorCmd_go
import termios
import tty

class G1HeadCtrlNode(Node):
    def __init__(self):
        super().__init__("ros2_g1_head_control_node")
        self.head_state_sub = self.create_subscription(MotorStates, "g1_comp_servo/state", self.head_state_cb, 1)
        self.head_state_sub
        self.head_cmd_pub = self.create_publisher(MotorCmds, "g1_comp_servo/cmd", 1)
        
        self.head_pitch_state = 0.0
        self.head_yaw_state = 0.0
        
        self.pitch_max = 100.0  # degrees
        self.pitch_min = -18.0  # degrees
        self.yaw_max = 45.0    # degrees
        self.yaw_min = -45.0   # degrees
    
    def head_state_cb(self, msg: MotorStates):
        self.head_pitch_state = msg.states[1].q
        self.head_yaw_state = msg.states[0].q
    
    def ctrl_head(self, tgt_pitch: float, tgt_yaw: float):
        
        if tgt_pitch > self.pitch_max:
            tgt_pitch = self.pitch_max
        if tgt_pitch < self.pitch_min:
            tgt_pitch = self.pitch_min
        if tgt_yaw > self.yaw_max:
            tgt_yaw = self.yaw_max
        if tgt_yaw < self.yaw_min:
            tgt_yaw = self.yaw_min
        
        head_cmd_msg = MotorCmds()
        pitch_motor_cmd = MotorCmd_go()
        pitch_motor_cmd.mode = 1  # position control mode
        pitch_motor_cmd.q = tgt_pitch
        pitch_motor_cmd.kp = 0.0
        pitch_motor_cmd.dq = 0.0
        pitch_motor_cmd.kd = 0.0
        pitch_motor_cmd.tau = 0.0

        yaw_motor_cmd = MotorCmd_go()
        yaw_motor_cmd.mode = 1  # position control mode
        yaw_motor_cmd.q = tgt_yaw
        yaw_motor_cmd.kp = 0.0
        yaw_motor_cmd.dq = 0.0
        yaw_motor_cmd.kd = 0.0
        yaw_motor_cmd.tau = 0.0

        head_cmd_msg.cmds.append(yaw_motor_cmd)
        head_cmd_msg.cmds.append(pitch_motor_cmd)

        self.head_cmd_pub.publish(head_cmd_msg)
        
    def get_head_state(self):
        return self.head_pitch_state, self.head_yaw_state

# ---- keyboard helper for non-blocking stdin ----
class KBHit:
    """A simple class to detect key presses on Unix-like terminals (non-blocking)."""
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_term = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
    
    def kbhit(self, timeout: float = 0.0) -> bool:
        """Return True if a key was hit within timeout seconds."""
        dr, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(dr)
    
    def getch(self) -> str:
        """Read a single character (no blocking if kbhit was True)."""
        return sys.stdin.read(1)
    
    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_term)

if __name__ == "__main__":
    rclpy.init()
    head_ctrl_node = G1HeadCtrlNode()
    tgt_pitch = 0.0
    tgt_yaw = 0.0
    head_ctrl_node.ctrl_head(0., 0.)
    step = 1.0
    
    kb = KBHit()
    head_ctrl_node.get_logger().info("Use W/S to increase/decrease pitch, A/D to decrease/increase yaw. Ctrl+C or ESC to quit.")
    
    try:
        # rate = head_ctrl_node.create_rate(50)  # 50 Hz
        while rclpy.ok():
            rclpy.spin_once(head_ctrl_node, timeout_sec=0.00)
            
            # If keyboard input, read and process
            if kb.kbhit(timeout=0.0):
                ch = kb.getch()
                # Handle common case inputs
                if ch in ('w', 'W'):
                    tgt_pitch += step
                elif ch in ('s', 'S'):
                    tgt_pitch -= step
                elif ch in ('a', 'A'):
                    tgt_yaw -= step
                elif ch in ('d', 'D'):
                    tgt_yaw += step
                elif ch == '\x1b':  # ESC key
                    head_ctrl_node.get_logger().info("ESC pressed, exiting.")
                    break
                
            head_ctrl_node.ctrl_head(tgt_pitch, tgt_yaw)
            
            pitch, yaw = head_ctrl_node.get_head_state()
            head_ctrl_node.get_logger().info(f"Cmd -> Pitch: {tgt_pitch:.3f}, Yaw: {tgt_yaw:.3f} | "
                        f"State -> Pitch: {pitch:.3f}, Yaw: {yaw:.3f}")
            # rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        kb.restore()
        head_ctrl_node.destroy_node()
        rclpy.shutdown()