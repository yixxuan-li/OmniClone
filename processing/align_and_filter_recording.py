import argparse
import json
import os
import numpy as np
import glob
import tqdm
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


def interpolate_array(array_data, old_times, new_times, kind='linear'):
    """
    Interpolate array data to new time points.
    
    Args:
        array_data: List of arrays, shape (n_frames, n_values)
        old_times: Old timestamps, shape (n_frames,)
        new_times: New timestamps, shape (n_samples,)
        kind: Interpolation method ('linear', 'cubic', etc.)
    
    Returns:
        List of interpolated arrays
    """
    if not array_data:
        return []
    
    array_data = np.array(array_data)
    interpolator = interp1d(old_times, array_data, axis=0, kind=kind, 
                           bounds_error=False, fill_value='extrapolate')
    interpolated = interpolator(new_times)
    return interpolated.tolist()


def interpolate_dict(data, old_times, new_times, kind='linear'):
    """
    Interpolate dictionary data to new time points.
    
    Args:
        data: List of dictionaries
        old_times: Old timestamps
        new_times: New timestamps
    
    Returns:
        List of interpolated dictionaries
    """
    if not data:
        return []
    
    # First pass: collect all keys from all frames
    all_keys = set()
    for frame in data:
        all_keys.update(frame.keys())
    
    interpolated_data = []
    
    for new_time in new_times:
        new_frame = {}
        
        for key in all_keys:
            # Skip timestamp - we'll set it to new_time
            if key == 'timestamp':
                new_frame[key] = new_time
                continue
            
            # Get values for this key across all frames
            values = []
            times = []
            for i, frame in enumerate(data):
                if key in frame:
                    values.append(frame[key])
                    times.append(old_times[i])
            
            if not values:
                continue
            
            # If it's a list/array, try to interpolate
            if isinstance(values[0], list) and len(values[0]) > 0:
                # Check if all values have the same length
                if all(len(v) == len(values[0]) for v in values):
                    try:
                        # Convert to numpy array and interpolate
                        arr_values = np.array(values)
                        interpolator = interp1d(times, arr_values, axis=0, kind=kind,
                                               bounds_error=False, fill_value='extrapolate')
                        interpolated_value = interpolator(new_time)
                        new_frame[key] = interpolated_value.tolist() if isinstance(interpolated_value, np.ndarray) else interpolated_value
                    except Exception:
                        # Fallback to nearest neighbor
                        idx = np.abs(np.array(times) - new_time).argmin()
                        new_frame[key] = values[idx]
                else:
                    # Variable length - use nearest neighbor
                    idx = np.abs(np.array(times) - new_time).argmin()
                    new_frame[key] = values[idx]
            elif isinstance(values[0], dict):
                # Nested dictionary - use nearest neighbor for now
                idx = np.abs(np.array(times) - new_time).argmin()
                new_frame[key] = values[idx]
            else:
                # Scalar values - interpolate
                try:
                    interpolator = interp1d(times, values, kind=kind,
                                           bounds_error=False, fill_value='extrapolate')
                    new_frame[key] = float(interpolator(new_time))
                except Exception:
                    # Fallback to nearest neighbor
                    idx = np.abs(np.array(times) - new_time).argmin()
                    new_frame[key] = values[idx]
        
        interpolated_data.append(new_frame)
    
    return interpolated_data


def normalize_quaternion(q):
    """
    Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion as list or numpy array [w, x, y, z] or [x, y, z, w]
    
    Returns:
        Normalized quaternion as list
    """
    q = np.array(q)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        # Return identity quaternion if quaternion is too small
        return np.array([1.0, 0.0, 0.0, 0.0])
    return (q / norm).tolist()


def normalize_quaternions_in_recordings(recordings):
    """
    Normalize all quaternions in recordings.
    
    Args:
        recordings: List of frame dictionaries
    
    Returns:
        List of recordings with normalized quaternions
    """
    for frame in recordings:
        # Normalize qpos quaternion (indices 3:7)
        if 'qpos' in frame and len(frame['qpos']) >= 7:
            qpos = frame['qpos']
            quat = qpos[3:7]
            normalized_quat = normalize_quaternion(quat)
            qpos[3:7] = normalized_quat
            frame['qpos'] = qpos
        
        # Normalize quaternions in rigid_bodies
        if 'rigid_bodies' in frame and isinstance(frame['rigid_bodies'], dict):
            for key, value in frame['rigid_bodies'].items():
                if isinstance(value, list) and len(value) >= 7:
                    # Format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
                    pos = value[:3]
                    quat = value[3:7]
                    normalized_quat = normalize_quaternion(quat)
                    frame['rigid_bodies'][key] = pos + normalized_quat
    
    return recordings


def apply_low_pass_filter(data, sample_rate, cutoff_freq, order=4):
    """
    Apply a low-pass filter using Butterworth filter to remove high-frequency noise.
    
    Args:
        data: Array of shape (n_samples, n_features)
        sample_rate: Sample rate in Hz
        cutoff_freq: Cutoff frequency in Hz
        order: Filter order (default: 4)
    
    Returns:
        Filtered data array
    """
    if len(data) < 3:
        return data
    
    nyquist = sample_rate / 2.0
    normal_cutoff = cutoff_freq / nyquist
    
    # Design the filter
    b, a = butter(order, normal_cutoff, btype='low')
    
    # Apply the filter bidirectionally to avoid phase shift
    filtered_data = filtfilt(b, a, data, axis=0)
    
    return filtered_data


def filter_recordings(recordings, sample_rate, cutoff_freq):
    """
    Apply low-pass filter to recordings to remove high-frequency noise.
    
    Args:
        recordings: List of frame dictionaries
        sample_rate: Sample rate in Hz
        cutoff_freq: Cutoff frequency in Hz
    
    Returns:
        List of filtered frame dictionaries
    """
    if len(recordings) < 3:
        return recordings
    
    print(f"Applying low-pass filter: cutoff={cutoff_freq}Hz, sample_rate={sample_rate}Hz")
    
    # Convert to a structured format for filtering
    filtered_recordings = []
    
    # Collect all keys
    all_keys = set()
    for frame in recordings:
        all_keys.update(frame.keys())
    
    # Process each key that contains numerical arrays
    keys_to_filter = []
    for key in all_keys:
        if key == 'timestamp':
            continue
        if recordings[0].get(key) is not None:
            if isinstance(recordings[0][key], list):
                if len(recordings[0][key]) > 0 and isinstance(recordings[0][key][0], (int, float)):
                    keys_to_filter.append(key)
    
    # Filter each array-valued key
    for key in keys_to_filter:
        # Extract the array data across all frames
        array_data = []
        for frame in recordings:
            if key in frame and frame[key] is not None:
                array_data.append(frame[key])
            else:
                array_data.append(None)
        
        # Find the first valid frame to get the array length
        valid_data = None
        for data in array_data:
            if data is not None:
                valid_data = data
                break
        
        if valid_data is None:
            continue
        
        # Only filter if all values are lists/arrays
        if isinstance(valid_data, list):
            try:
                arr = np.array([d if d is not None else [0]*len(valid_data) for d in array_data])
                
                # Apply filter
                if arr.shape[0] > 3:  # Need at least 3 samples for filtering
                    arr_filtered = apply_low_pass_filter(arr, sample_rate, cutoff_freq)
                    
                    # Write back to recordings
                    for i, frame in enumerate(recordings):
                        if key in frame:
                            recordings[i][key] = arr_filtered[i].tolist()
            except Exception as e:
                print(f"Warning: Could not filter key '{key}': {e}")
    
    # Filter rigid_bodies
    if len(recordings) > 3:
        # Collect all rigid_body keys
        all_rb_keys = set()
        for frame in recordings:
            if 'rigid_bodies' in frame and isinstance(frame['rigid_bodies'], dict):
                all_rb_keys.update(frame['rigid_bodies'].keys())
        
        for rb_key in all_rb_keys:
            # Extract the data for this rigid body
            rb_data = []
            for frame in recordings:
                if 'rigid_bodies' in frame and rb_key in frame['rigid_bodies']:
                    value = frame['rigid_bodies'][rb_key]
                    if isinstance(value, list) and len(value) == 7:
                        rb_data.append(value)
                    else:
                        rb_data.append(None)
                else:
                    rb_data.append(None)
            
            if not rb_data or all(d is None for d in rb_data):
                continue
            
            # Filter the data
            try:
                arr = np.array([d if d is not None else [0]*7 for d in rb_data])
                
                if arr.shape[0] > 3:
                    arr_filtered = apply_low_pass_filter(arr, sample_rate, cutoff_freq)
                    
                    # Write back to recordings
                    for i, frame in enumerate(recordings):
                        if 'rigid_bodies' in frame and rb_key in frame['rigid_bodies']:
                            if isinstance(frame['rigid_bodies'][rb_key], list):
                                frame['rigid_bodies'][rb_key] = arr_filtered[i].tolist()
            except Exception as e:
                print(f"Warning: Could not filter rigid body '{rb_key}': {e}")
    
    return recordings


def filter_out_invalid_datas(recordings):
    """
    Filter out invalid datas from recordings.
    
    Args:
        recordings: List of frame dictionaries
    """
    invalid_rb_names = set()
    invalid_rb_frame_count = dict()
    occurance_rb_frame_count = dict()

    for frame in recordings:
        if 'rigid_bodies' in frame and isinstance(frame['rigid_bodies'], dict):
            single_invalid_rb_names = set()
            for rb_key, rb_data in frame['rigid_bodies'].items():
                if rb_key not in occurance_rb_frame_count:
                    occurance_rb_frame_count[rb_key] = 0
                occurance_rb_frame_count[rb_key] += 1

                if isinstance(rb_data, list):
                    if len(rb_data) == 7:
                        frame['rigid_bodies'][rb_key] = rb_data
                    else:
                        invalid_rb_names.add(rb_key)
                        continue
                    if np.linalg.norm(np.array(rb_data[:3])) > 10000.0:
                        if rb_key not in invalid_rb_frame_count:
                            invalid_rb_frame_count[rb_key] = 0
                        invalid_rb_frame_count[rb_key] += 1
                        single_invalid_rb_names.add(rb_key)
                else:
                    invalid_rb_names.add(rb_key)
                    continue
            
            for rb_key in single_invalid_rb_names:
                frame['rigid_bodies'].pop(rb_key)

    for k, v in invalid_rb_frame_count.items():
        if v == occurance_rb_frame_count[k]:
            invalid_rb_names.add(k)

    for frame in recordings:
        if 'rigid_bodies' in frame and isinstance(frame['rigid_bodies'], dict):
            for key in invalid_rb_names:
                if key in frame['rigid_bodies']:
                    frame['rigid_bodies'].pop(key)
    return recordings


def align_recording(recordings, target_fps=50.0, interpolation='linear', apply_filter=False, cutoff_freq=10.0):
    """
    Align a recording to a fixed framerate.
    
    Args:
        recordings: List of frame dictionaries
        target_fps: Target framerate in Hz
        interpolation: Interpolation method ('linear', 'cubic', 'nearest')
        apply_filter: Whether to apply low-pass filter
        cutoff_freq: Cutoff frequency for low-pass filter
    
    Returns:
        List of aligned frame dictionaries
    """
    if len(recordings) < 2:
        return recordings

    # Filter out invalid datas
    recordings = filter_out_invalid_datas(recordings)

    # Extract timestamps
    old_times = np.array([frame.get('timestamp', 0.0) for frame in recordings])
    
    # Normalize timestamps to start from 0
    old_times = old_times - old_times[0]
    
    # Create new timestamps at target framerate
    duration = old_times[-1]
    dt = 1.0 / target_fps
    new_times = np.arange(0, duration + dt, dt)
    
    print(f"Original: {len(recordings)} frames over {duration:.2f}s ({duration/len(recordings):.4f}s per frame)")
    print(f"Aligned: {len(new_times)} frames over {duration:.2f}s at {target_fps}Hz")
    
    # Interpolate the data
    aligned_recordings = interpolate_dict(recordings, old_times, new_times, kind=interpolation)
    
    # Apply low-pass filter if requested
    if apply_filter:
        aligned_recordings = filter_recordings(aligned_recordings, target_fps, cutoff_freq)
    
    # Normalize all quaternions
    print("Normalizing quaternions...")
    aligned_recordings = normalize_quaternions_in_recordings(aligned_recordings)
    
    return aligned_recordings


def main(args):
    """
    Align recording to fixed framerate.
    
    Args:
        args: Command line arguments
            - recording_path: Path to input recording JSON file
            - output_path: Path to save aligned recording
            - target_fps: Target framerate (default: 60.0 Hz)
            - interpolation: Interpolation method (default: 'linear')
    """
    
    # Load the recording
    print(f"Loading recording from {args.recording_path}...")
    file_name = args.recording_path.split('/')[-1].split('.')[0]
    with open(args.recording_path, 'r') as f:
        recordings = json.load(f)
    
    if not isinstance(recordings, list):
        print("Error: Expected recordings to be a list of frames.")
        return
    
    print(f"Loaded {len(recordings)} frames")
    
    # Align the recording
    aligned_recordings = align_recording(
        recordings, 
        target_fps=args.target_fps,
        interpolation=args.interpolation,
        apply_filter=args.apply_filter,
        cutoff_freq=args.cutoff_freq
    )
    
    # Save the aligned recording
    print(f"Saving aligned recording to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, f'{file_name}.json'), 'w') as f:
        json.dump(aligned_recordings, f)
    
    print(f"Done! Aligned recording saved to {args.output_path}")
    print(f"Reduced from {len(recordings)} to {len(aligned_recordings)} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align recording to fixed framerate"
    )
    parser.add_argument(
        "--recording_path",
        type=str,
        required=True,
        help="Path to input recording JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datas/",
        help="Path to save aligned recording"
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=50.0,
        help="Target framerate in Hz (default: 50.0)"
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="linear",
        choices=["linear", "cubic", "nearest"],
        help="Interpolation method (default: 'linear')"
    )
    parser.add_argument(
        "--apply_filter",
        action="store_true",
        help="Apply low-pass filter to remove high-frequency noise"
    )
    parser.add_argument(
        "--cutoff_freq",
        type=float,
        default=10.0,
        help="Cutoff frequency for low-pass filter in Hz (default: 10.0)"
    )
    
    args = parser.parse_args()
    # resolve recording path
    recording_paths = glob.glob(args.recording_path)
    print(f"Found {len(recording_paths)} recordings")
    for recording_path in tqdm.tqdm(recording_paths):
        print(f"Processing {recording_path}...")
        args.recording_path = recording_path
        main(args)

