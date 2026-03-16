"""
Blender Script to Animate G1 Robot from JSON Motion Data

Usage:
1. Open Blender with your G1 robot model loaded
2. Scripting workspace -> Text Editor
3. Open this script file
4. Update the recording_path variable at the bottom
5. Run the script (Alt+P or click "Run Script")
"""

import json
import mathutils as mathutils


def get_joint_names():
    """Return the list of G1 joint names in qpos order."""
    return [
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
        "right_wrist_yaw_joint",
    ]


def animate_g1_robot(recordings, armature_obj, animate_root_transform=True):
    """
    Animate the G1 robot based on qpos data from recordings.
    
    Args:
        recordings: List of frame dictionaries with qpos data
        armature_obj: The G1 robot armature object
        animate_root_transform: Whether to animate root position on the armature object
    """
    import bpy
    
    if armature_obj is None:
        print("Error: No armature found")
        return
    
    # Ensure we start in OBJECT mode
    bpy.context.view_layer.objects.active = armature_obj
    
    # Try to set to OBJECT mode, but don't fail if we can't
    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception as e:
        print(f"Warning: Could not switch to OBJECT mode: {e}")
    
    # Get bones for later use  
    bones = armature_obj.pose.bones
    joint_names = get_joint_names()
    
    print(f"Found {len(bones)} bones in armature")
    print("Mapping joints to bones...")
    
    # Map joint names to bone names (with .revolute.bone suffix)
    joint_to_bone = {}
    for joint_name in joint_names:
        # Try different possible bone name formats
        possible_bone_names = [
            f"{joint_name}.revolute.bone",
            f"{joint_name}.bone",
            joint_name,
        ]
        
        for bone_name in possible_bone_names:
            if bone_name in bones:
                joint_to_bone[joint_name] = bone_name
                break
    
    # Find root bone
    root_bone_name = None
    root_keywords = ["pelvis", "base", "root", "world"]
    for bone_name in bones.keys():
        if any(keyword in bone_name.lower() for keyword in root_keywords):
            root_bone_name = bone_name
            break
    
    print(f"Found {len(joint_to_bone)} mapped joints")
    print(f"Root bone: {root_bone_name}")
    
    # Debug: Print first few frames
    if recordings and 'qpos' in recordings[0]:
        qpos = recordings[0]['qpos']
        print(f"First frame qpos shape: {len(qpos)}")
        print(f"Root pos: {qpos[:3]}")
        print(f"Root quat: {qpos[3:7]}")
        if len(qpos) > 7:
            print(f"First joint values: {qpos[7:12]}")
    
    # Ensure we start in OBJECT mode for location setting
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animate each frame
    for frame_idx, frame in enumerate(recordings):
        # Set current frame
        bpy.context.scene.frame_set(frame_idx + 1)
        
        if 'qpos' not in frame:
            continue
        
        qpos = frame['qpos']
        
        if len(qpos) < 7:
            continue
        
        # Root position (indices 0-2)
        root_pos = qpos[:3]
        
        # Root rotation quaternion (indices 3-6): [w, x, y, z]
        root_quat = qpos[3:7]
        
        # Joint positions (indices 7+)
        if len(qpos) > 7:
            joint_positions = qpos[7:]
        else:
            joint_positions = []
        
        # Animate rigid bodies if they exist (must be done in OBJECT mode)
        if 'rigid_bodies' in frame and frame['rigid_bodies']:
            animate_rigid_bodies(frame['rigid_bodies'], frame_idx, armature_obj)
        
        # Ensure armature is active before switching to POSE mode
        if bpy.context.view_layer.objects.active != armature_obj:
            bpy.context.view_layer.objects.active = armature_obj
        
        # Switch to POSE mode to animate bones
        try:
            if bpy.context.mode != 'POSE':
                bpy.ops.object.mode_set(mode='POSE')
        except Exception as e:
            print(f"Warning: Could not switch to POSE mode: {e}")
            print(f"Current mode: {bpy.context.mode}")
            print(f"Active object: {bpy.context.view_layer.objects.active}")
            # Continue anyway
        
        # Animate root bone (both location and rotation)
        assert root_bone_name is not None, "Root bone not found"
        if root_bone_name and root_bone_name in bones:
            root_bone = bones[root_bone_name]
            
            # Set root bone location (this is often more reliable than armature object location)
            if animate_root_transform:
                # Scale root position if needed
                SCALE_FACTOR = 100.0  # Adjust this if needed
                scaled_root_pos = [x * SCALE_FACTOR for x in root_pos]
                
                # Set root bone location
                root_bone.location = mathutils.Vector(scaled_root_pos)
                root_bone.keyframe_insert(data_path="location", frame=frame_idx + 1)
                
                # Debug first frame
                if frame_idx == 0:
                    print(f"Setting root bone location to: {root_pos}")
                    print(f"Scaled root location to: {scaled_root_pos}")
            
            # Set root rotation from quaternion
            quat_blender = mathutils.Quaternion((root_quat[0], root_quat[1], root_quat[2], root_quat[3]))
            # Normalize quaternion
            quat_blender.normalize()
            root_bone.rotation_quaternion = quat_blender
            root_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)
        
        # Animate individual joints
        for i, joint_name in enumerate(joint_names):
            if i >= len(joint_positions):
                break
            
            if joint_name not in joint_to_bone:
                continue
            
            bone_name = joint_to_bone[joint_name]
            if bone_name not in bones:
                continue
            
            bone = bones[bone_name]
            joint_value = joint_positions[i]
            
            # Check if bone uses quaternion or euler rotation
            if hasattr(bone, 'rotation_quaternion') and bone.rotation_mode == 'QUATERNION':
                # Convert angle to quaternion (rotation around local Y axis for most G1 joints)
                # Create identity quaternion and apply Y-axis rotation
                angle = joint_value
                quat = mathutils.Quaternion((0, 1, 0), angle)  # (axis, angle)
                bone.rotation_quaternion = quat
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)
            else:
                # Use euler rotation
                bone.rotation_euler[1] = joint_value  # Y rotation (pitch)
                bone.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_idx + 1)
        
        # Print progress
        if (frame_idx + 1) % 50 == 0:
            print(f"Animated frame {frame_idx + 1}/{len(recordings)}")


def animate_rigid_bodies(rigid_bodies_dict, frame_idx, armature_obj):
    """
    Animate rigid body objects in the scene.
    
    Args:
        rigid_bodies_dict: Dictionary of rigid body name to [pos, quat] data
        frame_idx: Current frame index
        armature_obj: The armature object (for context switching)
    """
    import bpy
    import mathutils
    
    # Ensure we're in OBJECT mode for creating objects
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Create or get the rigid bodies collection
    collection_name = "RigidBodies"
    if collection_name not in bpy.data.collections:
        rb_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(rb_collection)
        print(f"Created collection: {collection_name}")
    else:
        rb_collection = bpy.data.collections[collection_name]
    
    for rb_name, rb_data in rigid_bodies_dict.items():
        if not isinstance(rb_data, list) or len(rb_data) < 7:
            raise ValueError(f"Rigid body data for {rb_name} is invalid")
        
        # Parse data: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        pos = [x * 100.0 for x in rb_data[:3]]
        quat = rb_data[3:7]
        
        # Check if object exists, if not create it
        if rb_name not in bpy.data.objects:
            # Deselect all
            bpy.ops.object.select_all(action='DESELECT')
            
            # Create an empty object for visualization with ARROWS type
            bpy.ops.object.empty_add(type='ARROWS', location=pos)
            rb_obj = bpy.context.object
            rb_obj.name = rb_name
            
            # Make it more visible
            rb_obj.empty_display_type = 'ARROWS'
            rb_obj.empty_display_size = 20.0  # Size in Blender units
            rb_obj.show_in_front = True  # Show in front of other objects
            
            # Create a text label for the rigid body
            bpy.ops.object.text_add(location=(pos[0], pos[1], pos[2] + 0.2))  # Slightly above the rigid body
            text_obj = bpy.context.object
            text_obj.name = f"{rb_name}_label"
            text_obj.data.body = rb_name  # Set the text content
            text_obj.data.size = 10.0  # Small text size
            text_obj.data.align_x = 'CENTER'  # Center aligned
            text_obj.rotation_euler = (1.5708, 0, 0)  # Rotate 90 degrees to face camera (rough approximation)
            
            # Make text always face camera (billboard effect using constraints)
            # Note: We'll use a simple offset for now, proper billboard would need constraints
            text_obj.location = (pos[0], pos[1] + 0.2, pos[2] + 0.2)
            
            # Link text to rigid bodies collection
            for col in text_obj.users_collection:
                col.objects.unlink(text_obj)
            rb_collection.objects.link(text_obj)
            
            # Parent text to rigid body empty for synchronized movement
            bpy.ops.object.select_all(action='DESELECT')
            rb_obj.select_set(True)
            text_obj.select_set(True)
            bpy.context.view_layer.objects.active = rb_obj
            bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
            
            # Unlink from default collection and link to rigid bodies collection
            for col in rb_obj.users_collection:
                col.objects.unlink(rb_obj)
            rb_collection.objects.link(rb_obj)
            
            # Debug first created object
            if frame_idx == 0:
                print(f"Created rigid body object: {rb_name} at {pos}")
                print(f"  Empty display type: {rb_obj.empty_display_type}")
                print(f"  Empty display size: {rb_obj.empty_display_size}")
                print(f"  Added text label: {rb_name}")
                print(f"  Quaternion: {quat}")
        else:
            rb_obj = bpy.data.objects[rb_name]
            
            # Also log quaternion on first frame for existing objects
            if frame_idx == 0:
                print(f"Updating rigid body: {rb_name}")
                print(f"  Quaternion: {quat}")
        
        # Set position
        rb_obj.location = mathutils.Vector(pos)
        rb_obj.keyframe_insert(data_path="location", frame=frame_idx + 1)
        
        # Set rotation
        # Convert quaternion to rotation, handling coordinate systems
        # Data format: [quat_w, quat_x, quat_y, quat_z]
        # Blender quaternion: (w, x, y, z)
        
        try:
            rb_obj.rotation_mode = 'QUATERNION'
            quat_blender = mathutils.Quaternion((quat[0], quat[1], quat[2], quat[3]))
            quat_blender.normalize()
            rb_obj.rotation_quaternion = quat_blender
            rb_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)
        except:
            # Fallback to euler if quaternion doesn't work
            rb_obj.rotation_mode = 'XYZ'
            quat_blender = mathutils.Quaternion((quat[0], quat[1], quat[2], quat[3]))
            quat_blender.normalize()
            euler = quat_blender.to_euler('XYZ')
            rb_obj.rotation_euler = euler
            rb_obj.keyframe_insert(data_path="rotation_euler", frame=frame_idx + 1)


def clear_existing_animations(armature_obj):
    """
    Clear all existing animations from the armature and rigid bodies.
    
    Args:
        armature_obj: The armature object to clear animations from
    """
    import bpy
    
    print("Clearing existing animations...")
    
    # Clear animation data from armature object
    if armature_obj and armature_obj.animation_data:
        action = armature_obj.animation_data.action
        if action:
            bpy.data.actions.remove(action)
        armature_obj.animation_data_clear()
    
    # Clear animation data from bones
    if armature_obj and hasattr(armature_obj, 'pose'):
        bpy.context.view_layer.objects.active = armature_obj
        try:
            bpy.ops.object.mode_set(mode='POSE')
            # Select all bones and clear their keyframes
            for bone in armature_obj.pose.bones:
                bone.bone.select = True
            # Clear all bone keyframes
            bpy.ops.anim.keyframe_delete(type='ALL', confirm_success=False)
        except:
            pass  # If we can't clear, continue anyway
    
    # Clear animation data from rigid body objects
    collection_name = "RigidBodies"
    if collection_name in bpy.data.collections:
        rb_collection = bpy.data.collections[collection_name]
        for obj in rb_collection.objects:
            if obj.animation_data:
                action = obj.animation_data.action
                if action:
                    bpy.data.actions.remove(action)
                obj.animation_data_clear()
    
    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    print("Animations cleared.")


def import_json_to_blender(json_path):
    """
    Import a JSON recording file and animate the G1 robot.
    
    Args:
        json_path: Path to the JSON recording file
    """
    import bpy
    
    print(f"Loading recording from {json_path}...")
    
    with open(json_path, 'r') as f:
        recordings = json.load(f)
    
    if not isinstance(recordings, list):
        print("Error: Expected recordings to be a list of frames.")
        return
    
    print(f"Loaded {len(recordings)} frames")
    
    # Calculate framerate from timestamps
    if len(recordings) > 1 and 'timestamp' in recordings[0]:
        timestamps = [frame.get('timestamp', 0) for frame in recordings if 'timestamp' in frame]
        if len(timestamps) > 1:
            # Calculate average time delta
            deltas = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_delta = sum(deltas) / len(deltas)
            # Calculate framerate (inverse of delta)
            calculated_fps = 1.0 / avg_delta
            print(f"Calculated framerate: {calculated_fps:.2f} fps (avg delta: {avg_delta:.4f}s)")
            
            # Set the scene fps
            bpy.context.scene.render.fps = int(round(calculated_fps))
            bpy.context.scene.render.fps_base = 1.0
            print(f"Set Blender scene fps to: {calculated_fps:.2f}")
        else:
            print("Warning: Could not calculate framerate from timestamps, using default 60 fps")
    else:
        print("No timestamps found in recording, using default 60 fps")
    
    # Find the G1 robot in the scene
    g1_robot = find_g1_robot()
    
    if g1_robot is None:
        print("Error: G1 robot not found in the scene!")
        print("Please make sure your G1 robot armature is loaded in Blender.")
        return
    
    # Clear existing animations before importing new motion
    clear_existing_animations(g1_robot)
    
    # Set timeline
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(recordings)
    
    # Animate the robot based on qpos data
    print("Animating G1 robot based on qpos data...")
    animate_g1_robot(recordings, g1_robot)
    
    print(f"Done! Animated {len(recordings)} frames")
    print("\nTips:")
    print("- Press Space to play the animation")
    print("- Press Alt+A to play/pause")
    print("- Adjust frame range in Timeline")


def find_g1_robot():
    """
    Find the G1 robot armature in the Blender scene.
    
    Returns:
        The G1 armature object or None if not found
    """
    import bpy
    
    # Try common names for G1 robot
    possible_names = ["G1", "g1", "unitree_g1", "robot", "Armature", "g1_robot"]
    
    for obj_name, obj in bpy.data.objects.items():
        if obj.type == 'ARMATURE':
            # Check if it looks like a G1 robot
            if any(name.lower() in obj.name.lower() for name in possible_names):
                print(f"Found G1 robot: {obj.name}")
                return obj
    
    # If no specific match, just return the first armature
    for obj_name, obj in bpy.data.objects.items():
        if obj.type == 'ARMATURE':
            print(f"Found armature: {obj.name} (assuming it's the G1 robot)")
            return obj
    
    print("Warning: G1 robot not found in scene!")
    return None


if __name__ == "__main__":
    # MODIFY THIS PATH TO YOUR JSON FILE
    recording_path = "/home/whx/project/jingze/ANY2Humanoid/datas/recordings_20251124_233912.json"
    
    import_json_to_blender(recording_path)
