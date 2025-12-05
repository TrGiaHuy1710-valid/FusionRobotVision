import socket
import json
import threading
import time
import numpy as np
from queue import Queue
import h5py
import uuid
import os


class UDPClient:
    def __init__(self, ip="localhost", port=3939, dev_mode=False, dev_file="data_0.json"):
        self.dev_mode = dev_mode
        self.dev_file = dev_file
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.ip, self.port))
        self.lock = threading.Lock()
        self.queue = Queue()
        self.running = True
        self.file_count = 0
        self.batch_data = []
        self.cached_data = None  # Add cache for last received data
        self.is_recording = False
        self.recording_paused = False
        self.current_filename = None
        self.batch_data = []
        self.h5_file = None
        self.frame_count = 0

    def run(self):
        if self.dev_mode:
            # Check if the dev file is an HDF5 file
            if self.dev_file.endswith('.h5'):
                try:
                    with h5py.File(self.dev_file, 'r') as f:
                        print(f"Simulating data from HDF5 file: {self.dev_file}")
                        frame_count = len(f['frames'])
                        frame_idx = 0

                        # Store start time to calculate relative playback timing
                        start_abs_time = time.time()
                        first_frame_time = None

                        while self.running:
                            if frame_idx >= frame_count:
                                frame_idx = 0
                                print("Looping HDF5 data...")
                                start_abs_time = time.time()
                                first_frame_time = None

                            # Get the frame data and convert to JSON-compatible structure
                            frame_name = f"frame_{frame_idx}"
                            if frame_name in f['frames']:
                                frame_data = {}
                                frame_data['scene'] = {'actors': []}

                                # Get timestamps
                                abs_timestamp = None
                                rel_timestamp = None

                                if 'abs_timestamp' in f['frames'][frame_name].attrs:
                                    abs_timestamp = float(f['frames'][frame_name].attrs['abs_timestamp'])

                                if 'rel_timestamp' in f['frames'][frame_name].attrs:
                                    rel_timestamp = float(f['frames'][frame_name].attrs['rel_timestamp'])
                                elif 'timestamp' in f['frames'][frame_name].attrs:
                                    rel_timestamp = float(f['frames'][frame_name].attrs['timestamp'])

                                # Set timestamp in the data
                                if rel_timestamp is not None:
                                    frame_data['scene']['timestamp'] = rel_timestamp

                                # If we have absolute timestamps, use them to pace playback
                                if abs_timestamp is not None:
                                    if first_frame_time is None:
                                        first_frame_time = abs_timestamp

                                    # Calculate delay based on original capture timing
                                    frame_delay = abs_timestamp - first_frame_time
                                    curr_playback_time = time.time() - start_abs_time

                                    # Sleep to maintain original timing between frames
                                    if curr_playback_time < frame_delay:
                                        time.sleep(frame_delay - curr_playback_time)

                                # Get actor data
                                for actor_name in f['frames'][frame_name]:
                                    if actor_name.startswith('actor_'):
                                        actor_data = {}
                                        actor_group = f['frames'][frame_name][actor_name]

                                        # Get actor metadata
                                        for attr_name, attr_value in actor_group.attrs.items():
                                            actor_data[attr_name] = attr_value

                                        # Get body data
                                        if 'body' in actor_group:
                                            body_data = {}
                                            for part_name in actor_group['body']:
                                                part_data = {}
                                                part_group = actor_group['body'][part_name]

                                                # Convert position array back to dict
                                                if 'position' in part_group:
                                                    pos = part_group['position'][()]
                                                    part_data['position'] = {'x': float(pos[0]), 'y': float(pos[1]),
                                                                             'z': float(pos[2])}

                                                # Convert rotation array back to dict
                                                if 'rotation' in part_group:
                                                    rot = part_group['rotation'][()]
                                                    part_data['rotation'] = {'x': float(rot[0]), 'y': float(rot[1]),
                                                                             'z': float(rot[2]), 'w': float(rot[3])}

                                                body_data[part_name] = part_data

                                            actor_data['body'] = body_data

                                        frame_data['scene']['actors'].append(actor_data)

                                # Convert to a flat JSON string and add to queue
                                text_data = json.dumps(frame_data, separators=(',', ':'))
                                with self.lock:
                                    self.queue.put(text_data)

                            frame_idx += 1
                            time.sleep(1 / 30)  # 30fps playback

                except Exception as e:
                    print(f"Error reading HDF5 file: {e}")
                    # Fall back to JSON if there's an issue
                    print("Falling back to JSON file")
                    self.run_json_dev_mode()
            else:
                # Fall back to the original JSON loading
                self.run_json_dev_mode()
        else:
            # Real UDP data receiving
            while self.running:
                try:
                    data, _ = self.sock.recvfrom(65507)  # Buffer size
                    if data:
                        text_data = data.decode('utf-8')
                        with self.lock:
                            self.queue.put(text_data)
                except:
                    break

    def run_json_dev_mode(self):
        """Run in dev mode with JSON file (original implementation)"""
        try:
            with open(self.dev_file, "r") as f:
                simulated_data = json.load(f)
            idx = 0
            print(f"Simulating data from JSON file: {self.dev_file}")
            while self.running:
                if idx >= len(simulated_data):
                    idx = 0
                    print(f"Looping JSON data...")
                item = simulated_data[idx]
                # Convert to a flat JSON string
                text_data = json.dumps(item, separators=(",", ":"))
                with self.lock:
                    self.queue.put(text_data)
                idx += 1
                time.sleep(1 / 30)  # 30fps playback
        except Exception as e:
            print(f"Error reading JSON file: {e}")

    def update_cache(self):
        """Update cached data from queue without removing it"""
        with self.lock:
            if not self.queue.empty():
                self.cached_data = self.queue.get()
                return self.cached_data
        return None

    def get_data(self):
        """Get cached data without clearing queue"""
        return self.cached_data

    def start_recording(self, filename):
        """Start recording data to an HDF5 file"""
        # Change extension to .h5
        base_name = os.path.splitext(filename)[0]
        self.current_filename = f"{base_name}.h5"

        # Create or open HDF5 file
        self.h5_file = h5py.File(self.current_filename, 'w')

        # Create main groups
        self.h5_file.create_group('metadata')
        self.h5_file.create_group('frames')

        # Store metadata
        self.h5_file['metadata'].attrs['start_time'] = time.time()
        self.h5_file['metadata'].attrs['date'] = time.strftime('%Y-%m-%d %H:%M:%S')

        self.is_recording = True
        self.recording_paused = False
        self.frame_count = 0
        print(f"Started recording to {self.current_filename}")

    def pause_recording(self):
        if self.is_recording:
            self.recording_paused = True
            # Flush data to disk
            if self.h5_file:
                self.h5_file.flush()
            print("Recording paused")

    def resume_recording(self):
        if self.is_recording:
            self.recording_paused = False
            print("Recording resumed")

    def stop_recording(self):
        if self.is_recording:
            if self.h5_file:
                # Update metadata with final frame count
                self.h5_file['metadata'].attrs['total_frames'] = self.frame_count
                self.h5_file['metadata'].attrs['end_time'] = time.time()

                # Close the file
                self.h5_file.close()
                self.h5_file = None
                print(f"Saved {self.frame_count} frames to {self.current_filename}")

        self.is_recording = False
        self.recording_paused = False
        self.frame_count = 0
        self.current_filename = None

    def save_data_batch(self, threshold=200):
        """Save incoming data to the HDF5 file"""
        data_item = self.update_cache()
        if data_item and self.is_recording and not self.recording_paused:
            try:
                # Parse JSON data
                json_data = json.loads(data_item)

                # Extract key data for efficient storage
                if self.h5_file:
                    # Create a frame group for this frame
                    frame_id = f"frame_{self.frame_count}"
                    frame_group = self.h5_file['frames'].create_group(frame_id)

                    # Store absolute timestamp (system time) for later alignment
                    frame_group.attrs['abs_timestamp'] = time.time()

                    # Store relative timestamp from the motion capture system
                    if 'scene' in json_data and 'timestamp' in json_data['scene']:
                        frame_group.attrs['rel_timestamp'] = json_data['scene']['timestamp']

                    # Store actor data
                    if 'scene' in json_data and 'actors' in json_data['scene']:
                        for i, actor in enumerate(json_data['scene']['actors']):
                            actor_group = frame_group.create_group(f"actor_{i}")

                            # Store actor metadata
                            if 'name' in actor:
                                actor_group.attrs['name'] = actor['name']
                            if 'meta' in actor:
                                for key, value in actor['meta'].items():
                                    actor_group.attrs[key] = value

                            # Store body data efficiently
                            if 'body' in actor:
                                body_group = actor_group.create_group('body')
                                for part_name, part_data in actor['body'].items():
                                    part_group = body_group.create_group(part_name)

                                    # Store position as a compact dataset
                                    if 'position' in part_data:
                                        pos = part_data['position']
                                        part_group.create_dataset('position', data=[pos['x'], pos['y'], pos['z']])

                                    # Store rotation as a compact dataset
                                    if 'rotation' in part_data:
                                        rot = part_data['rotation']
                                        part_group.create_dataset('rotation',
                                                                  data=[rot['x'], rot['y'], rot['z'], rot['w']])

                    # Increment frame counter
                    self.frame_count += 1

                    # Periodically flush to ensure data is written
                    if self.frame_count % 100 == 0:
                        self.h5_file.flush()
            except Exception as e:
                print(f"Error saving data: {e}")

    def close(self):
        self.stop_recording()  # Ensure recording is properly stopped
        self.running = False
        self.sock.close()

    def align_timestamps(self, target_file, output_file=None):
        """
        Aligns timestamps between two H5 recordings.

        Args:
            target_file (str): Path to target H5 file for alignment
            output_file (str, optional): Path for aligned output file. If None, returns alignment info.

        Returns:
            Dictionary with alignment info or None if output_file is specified
        """
        if not self.current_filename or not self.current_filename.endswith('.h5'):
            print("No active HDF5 recording file.")
            return None

        try:
            # Open our recording and target file
            with h5py.File(self.current_filename, 'r') as src_file, h5py.File(target_file, 'r') as tgt_file:
                # Get timestamps from both files
                src_timestamps = []
                tgt_timestamps = []

                for frame_name in src_file['frames']:
                    if 'abs_timestamp' in src_file['frames'][frame_name].attrs:
                        src_timestamps.append(src_file['frames'][frame_name].attrs['abs_timestamp'])

                for frame_name in tgt_file['frames']:
                    if 'abs_timestamp' in tgt_file['frames'][frame_name].attrs:
                        tgt_timestamps.append(tgt_file['frames'][frame_name].attrs['abs_timestamp'])

                # Calculate time offset between recordings
                if src_timestamps and tgt_timestamps:
                    time_offset = min(src_timestamps) - min(tgt_timestamps)

                    # Apply offset if output file specified
                    if output_file:
                        with h5py.File(output_file, 'w') as out_file:
                            # Copy metadata
                            out_file.create_group('metadata')
                            for attr_name, attr_value in src_file['metadata'].attrs.items():
                                out_file['metadata'].attrs[attr_name] = attr_value

                            # Add alignment metadata
                            out_file['metadata'].attrs['aligned_to'] = os.path.basename(target_file)
                            out_file['metadata'].attrs['time_offset'] = time_offset

                            # Create frames group
                            out_file.create_group('frames')

                            # Copy frames with adjusted timestamps
                            for frame_name in src_file['frames']:
                                # Create new frame group
                                src_frame = src_file['frames'][frame_name]
                                out_frame = out_file['frames'].create_group(frame_name)

                                # Copy attributes with adjusted timestamp
                                for attr_name, attr_value in src_frame.attrs.items():
                                    if attr_name == 'abs_timestamp':
                                        out_frame.attrs[attr_name] = attr_value - time_offset
                                    else:
                                        out_frame.attrs[attr_name] = attr_value

                                # Copy data recursively
                                self._copy_h5_group(src_frame, out_frame)

                        print(f"Aligned data written to {output_file}")
                        return None

                    return {
                        'time_offset': time_offset,
                        'src_start': min(src_timestamps),
                        'src_end': max(src_timestamps),
                        'tgt_start': min(tgt_timestamps),
                        'tgt_end': max(tgt_timestamps),
                    }

                print("Couldn't find timestamps in one or both files.")
                return None

        except Exception as e:
            print(f"Error aligning timestamps: {e}")
            return None

    def _copy_h5_group(self, src_group, dst_group):
        """Helper function to recursively copy HDF5 group structure"""
        for key in src_group.keys():
            if isinstance(src_group[key], h5py.Group):
                # Copy subgroup
                new_group = dst_group.create_group(key)
                # Copy attributes
                for attr_name, attr_value in src_group[key].attrs.items():
                    new_group.attrs[attr_name] = attr_value
                # Recursively copy content
                self._copy_h5_group(src_group[key], new_group)
            else:
                # Copy dataset
                dst_group.create_dataset(key, data=src_group[key][()])


class MotionSuitReceiver:
    def __init__(self, sources=[(3939,)], dev_mode=False, dev_file="data_0.json"):
        # Create multiple clients if needed
        self.clients = []
        for s in sources:
            client = UDPClient(port=s[0], dev_mode=dev_mode, dev_file=dev_file)
            th = threading.Thread(target=client.run, daemon=True)
            th.start()
            self.clients.append(client)

    def get_latest_data(self):
        """
        Return a dictionary with the latest data from all clients.
        """
        results = {}
        for idx, c in enumerate(self.clients):
            c.save_data_batch()  # This will update the cache
            data = c.get_data()  # Get from cache instead of queue
            if data:
                try:
                    parsed = json.loads(data)
                    results[f"client_{idx}"] = parsed
                    # print(f"Received data from client {idx}")
                except:
                    pass
        return results

    def start_recording(self, filename):
        """Start recording to an HDF5 file for all clients"""
        # Create the directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        for client in self.clients:
            client.start_recording(filename)

    def pause_recording(self):
        for client in self.clients:
            client.pause_recording()

    def resume_recording(self):
        for client in self.clients:
            client.resume_recording()

    def stop_recording(self):
        for client in self.clients:
            client.stop_recording()

    def is_recording(self):
        return any(c.is_recording for c in self.clients)

    def is_recording_paused(self):
        return any(c.recording_paused for c in self.clients)

    def close(self):
        self.stop_recording()
        for c in self.clients:
            c.close()


class MocapDataLoader:
    """Utility class to load and process motion capture data from HDF5 files"""

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.frame_count = 0
        self.abs_timestamps = []
        self.rel_timestamps = []

    def open(self):
        """Open the HDF5 file and extract basic metadata"""
        try:
            self.file = h5py.File(self.filename, 'r')
            self.frame_count = len(self.file['frames'])

            # Extract all timestamps
            self.abs_timestamps = []
            self.rel_timestamps = []

            for i in range(self.frame_count):
                frame_name = f"frame_{i}"
                if frame_name in self.file['frames']:
                    if 'abs_timestamp' in self.file['frames'][frame_name].attrs:
                        self.abs_timestamps.append(self.file['frames'][frame_name].attrs['abs_timestamp'])
                    if 'rel_timestamp' in self.file['frames'][frame_name].attrs:
                        self.rel_timestamps.append(self.file['frames'][frame_name].attrs['rel_timestamp'])

            return True
        except Exception as e:
            print(f"Error opening file: {e}")
            return False

    def close(self):
        """Close the HDF5 file"""
        if self.file:
            self.file.close()
            self.file = None

    def get_frame(self, frame_idx):
        """Get a specific frame by index"""
        if not self.file:
            return None

        frame_name = f"frame_{frame_idx}"
        if frame_name in self.file['frames']:
            return self._convert_frame_to_dict(self.file['frames'][frame_name])
        return None

    def get_frame_at_time(self, timestamp, use_abs=True):
        """Get the frame closest to the specified timestamp"""
        if not self.file or not self.abs_timestamps:
            return None

        timestamps = self.abs_timestamps if use_abs else self.rel_timestamps
        if not timestamps:
            return None

        # Find the closest timestamp
        closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - timestamp))
        return self.get_frame(closest_idx)

    def get_body_part_trajectory(self, part_name, actor_idx=0):
        """
        Extract the trajectory of a specific body part across all frames
        Returns: Dictionary with 'positions', 'rotations', and 'timestamps'
        """
        if not self.file:
            return None

        positions = []
        rotations = []
        timestamps = []

        for i in range(self.frame_count):
            frame_name = f"frame_{i}"
            if frame_name in self.file['frames']:
                frame = self.file['frames'][frame_name]
                actor_name = f"actor_{actor_idx}"

                if actor_name in frame and 'body' in frame[actor_name] and part_name in frame[actor_name]['body']:
                    part_data = frame[actor_name]['body'][part_name]

                    if 'position' in part_data:
                        pos = part_data['position'][()]
                        positions.append({'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])})

                    if 'rotation' in part_data:
                        rot = part_data['rotation'][()]
                        rotations.append(
                            {'x': float(rot[0]), 'y': float(rot[1]), 'z': float(rot[2]), 'w': float(rot[3])})

                    if 'abs_timestamp' in frame.attrs:
                        timestamps.append(frame.attrs['abs_timestamp'])

        return {
            'positions': positions,
            'rotations': rotations,
            'timestamps': timestamps
        }

    def _convert_frame_to_dict(self, frame_group):
        """Convert an HDF5 frame group to a Python dictionary"""
        result = {}

        # Extract frame attributes
        for attr_name, attr_value in frame_group.attrs.items():
            result[attr_name] = attr_value

        # Extract actor data
        result['actors'] = []
        for actor_name in frame_group:
            if actor_name.startswith('actor_'):
                actor_data = {}
                actor_group = frame_group[actor_name]

                # Extract actor attributes
                for attr_name, attr_value in actor_group.attrs.items():
                    actor_data[attr_name] = attr_value

                # Extract body data if present
                if 'body' in actor_group:
                    body_data = {}
                    for part_name in actor_group['body']:
                        part_data = {}
                        part_group = actor_group['body'][part_name]

                        # Extract position and rotation
                        if 'position' in part_group:
                            pos = part_group['position'][()]
                            part_data['position'] = {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])}

                        if 'rotation' in part_group:
                            rot = part_group['rotation'][()]
                            part_data['rotation'] = {'x': float(rot[0]), 'y': float(rot[1]), 'z': float(rot[2]),
                                                     'w': float(rot[3])}

                        body_data[part_name] = part_data

                    actor_data['body'] = body_data

                result['actors'].append(actor_data)

        return result
