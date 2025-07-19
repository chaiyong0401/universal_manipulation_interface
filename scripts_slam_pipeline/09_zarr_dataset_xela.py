import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import zarr
import numpy as np
from tqdm import tqdm
import click
import shutil

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

# === [Configurable] ===
zip_path = "/home/embodied-ai/mcy/universal_manipulation_interface/xela_2/dataset.zarr_normalized.zarr.zip"
save_dir = "/home/embodied-ai/mcy/universal_manipulation_interface/xela_2/dataset_zarr_normalized_replaybuffer.zarr"

# === [Step 1] Load source zip zarr ===
store = zarr.ZipStore(zip_path, mode='r')
dataset = zarr.open(store, mode='r')

print(dataset.tree())

# === [Step 2] Read all data ===
camera0_rgb = dataset['data/camera0_rgb'][:]
camera0_tactile_normalized = dataset['data/camera0_tactile_normalized'][:]
camera0_tactile_offset = dataset['data/camera0_tactile_offset']

robot0_absolute_action = dataset['data/robot0_absolute_action'][:]
# robot0_absolute_action_4d = dataset['data/robot0_absolute_action_4d'][:] # 07/06
robot0_demo_start_pose = dataset['data/robot0_demo_start_pose'][:]
robot0_demo_end_pose = dataset['data/robot0_demo_end_pose'][:]
robot0_eef_pos = dataset['data/robot0_eef_pos'][:]
robot0_eef_rot_axis_angle = dataset['data/robot0_eef_rot_axis_angle'][:]
robot0_gripper_width = dataset['data/robot0_gripper_width'][:]
robot0_eef_pose_9d = dataset['data/robot0_eef_pose_9d'][:]

episode_ends = dataset['meta/episode_ends'][:]

print("✅ Loaded all source data.")

wrench_indices = [0, 3, 5, 10, 12, 15]  # 07/05 tactile 6d
camera0_tactile_offset_selected = camera0_tactile_offset[:, wrench_indices]

# === [Step 3] Prepare target ReplayBuffer format ===
if os.path.exists(save_dir):
    print("⚡️ Removing existing save_dir...")
    shutil.rmtree(save_dir)

root = zarr.group(save_dir)
zarr_data = root.create_group('data')
zarr_meta = root.create_group('meta')

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

# Save datasets for rdp venv format
zarr_data.create_dataset('left_wrist_img', data=camera0_rgb, chunks=(100, 224, 224, 3), dtype='uint8', compressor=compressor)
zarr_data.create_dataset('left_robot_normalized_offset', data=camera0_tactile_normalized, chunks=(1000, 16), dtype='float32', compressor=compressor)
# zarr_data.create_dataset('left_robot_tcp_wrench', data=camera0_tactile_offset, chunks=(1000, 16), dtype='float32', compressor=compressor)
zarr_data.create_dataset('left_robot_tcp_wrench', data=camera0_tactile_offset_selected, chunks=(1000, 6), dtype='float32', compressor=compressor) #07/05
zarr_data.create_dataset('left_robot_rot_axis_angle', data=robot0_eef_rot_axis_angle, chunks=(1000, 3), dtype='float32', compressor=compressor)
zarr_data.create_dataset('left_robot_gripper_width', data=robot0_gripper_width, chunks=(1000, 1), dtype='float32', compressor=compressor)

# tcp pose 9d, action 10d
zarr_data.create_dataset('left_robot_tcp_pos', data=robot0_eef_pos, chunks=(1000, 3), dtype='float32', compressor=compressor)
zarr_data.create_dataset('left_robot_tcp_pose', data=robot0_eef_pose_9d, chunks=(1000, 9), dtype='float32', compressor=compressor)
# zarr_data.create_dataset('left_robot_tcp_pose', data=robot0_eef_pos, chunks=(1000, 3), dtype='float32', compressor=compressor) # 07/06

zarr_data.create_dataset('action', data=robot0_absolute_action, chunks=(1000, 10), dtype='float32', compressor=compressor)
# zarr_data.create_dataset('action', data=robot0_absolute_action_4d, chunks=(1000, 4), dtype='float32', compressor=compressor) # 07/06
zarr_data.create_dataset('target', data=robot0_absolute_action, chunks=(1000, 10), dtype='float32', compressor=compressor)

zarr_data.create_dataset('robot0_demo_start_pose', data=robot0_demo_start_pose, chunks=(1000, 6), dtype='float64', compressor=compressor)
zarr_data.create_dataset('robot0_demo_end_pose', data=robot0_demo_end_pose, chunks=(1000, 6), dtype='float64', compressor=compressor)

zarr_meta.create_dataset('episode_ends', data=episode_ends, chunks=(1000,), dtype='int64')

print("✅ Done saving ReplayBuffer zarr format!")
print(zarr_data.tree())
