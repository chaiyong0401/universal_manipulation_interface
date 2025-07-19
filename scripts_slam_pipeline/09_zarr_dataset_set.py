import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import zarr
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import click
import shutil

from umi.common.pose_util import pose_to_mat, mat_to_pose10d
from scipy.spatial.transform import Rotation as R
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()
# # zip 파일 열기
# z = zarr.open("tactile_4/dataset.zarr_pca.zarr.zip", mode='r')

# print(z.tree())


# # 예를 들어 data group의 action dataset에서 일부 읽기
# action_data = z['data']['robot0_absolute_action']

# # 전체 shape
# print("Action dataset shape:", action_data.shape)

# # 처음 5개만 읽기
# print("First 5 actions:")
# print(action_data[:5])

# # 안전하게 디렉토리로 복사하기
# zarr.copy_store(z.store, zarr.DirectoryStore('/home/embodied-ai/mcy/universal_manipulation_interface/tactile_4/dataset_zarr_pca.zarr'), if_exists='replace')

# print("Copy completed!")


# === [Configurable] ===
zip_path = "/home/embodied-ai/mcy/universal_manipulation_interface/tactile_4/dataset.zarr_pca.zarr.zip"
save_dir = "/home/embodied-ai/mcy/universal_manipulation_interface/tactile_4/dataset_zarr_pca_replaybuffer.zarr"

# === [Step 1] Load source zip zarr ===
store = zarr.ZipStore(zip_path, mode='r')
dataset = zarr.open(store, mode='r')

print(dataset.tree())

# === [Step 2] Read all data ===
camera0_rgb = dataset['data/camera0_rgb'][:]
camera0_pca_emb = dataset['data/camera0_pca_emb'][:]
camera0_tactile_rgb = dataset['data/camera0_tactile_rgb'][:]
camera0_tactile_offset = dataset['data/camera0_tactile_offset'][:]

robot0_absolute_action = dataset['data/robot0_absolute_action'][:]
# robot0_absolute_action_4d = dataset['data/robot0_absolute_action_4d'][:]
robot0_demo_start_pose = dataset['data/robot0_demo_start_pose'][:]
robot0_demo_end_pose = dataset['data/robot0_demo_end_pose'][:]
robot0_eef_pos = dataset['data/robot0_eef_pos'][:]
robot0_eef_rot_axis_angle = dataset['data/robot0_eef_rot_axis_angle'][:]
robot0_gripper_width = dataset['data/robot0_gripper_width'][:]
robot0_eef_pose_9d = dataset['data/robot0_eef_pose_9d'][:]

episode_ends = dataset['meta/episode_ends'][:]

print("✅ Loaded all source data.")

# === [Step 3] Prepare target ReplayBuffer format ===
if os.path.exists(save_dir):
    print("⚡️ Removing existing save_dir...")
    shutil.rmtree(save_dir)

root = zarr.group(save_dir)
zarr_data = root.create_group('data')
zarr_meta = root.create_group('meta')

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

# Save datasets
zarr_data.create_dataset('left_wrist_img', data=camera0_rgb, chunks=(100, 224, 224, 3), dtype='uint8', compressor=compressor)
zarr_data.create_dataset('left_gripper1_marker_offset_emb', data=camera0_pca_emb, chunks=(1000, 15), dtype='float32', compressor=compressor)
zarr_data.create_dataset('left_gripper1_img', data=camera0_tactile_rgb, chunks=(100, 320, 240, 3), dtype='uint8', compressor=compressor)
zarr_data.create_dataset('left_gripper1_marker_offset', data=camera0_tactile_offset, chunks=(1000, 63, 1), dtype='uint8', compressor=compressor)
zarr_data.create_dataset('left_robot_rot_axis_angle', data=robot0_eef_rot_axis_angle, chunks=(1000, 3), dtype='float32', compressor=compressor)
zarr_data.create_dataset('left_robot_gripper_width', data=robot0_gripper_width, chunks=(1000, 1), dtype='float32', compressor=compressor)

# tcp pose 9d, action 10d
zarr_data.create_dataset('left_robot_tcp_pos', data=robot0_eef_pos, chunks=(1000, 3), dtype='float32', compressor=compressor)
zarr_data.create_dataset('left_robot_tcp_pose', data=robot0_eef_pose_9d, chunks=(1000, 9), dtype='float32', compressor=compressor)

zarr_data.create_dataset('action', data=robot0_absolute_action, chunks=(1000, 10), dtype='float32', compressor=compressor)
zarr_data.create_dataset('target', data=robot0_absolute_action, chunks=(1000, 10), dtype='float32', compressor=compressor)


# tcp pose 3d, action 4d
# zarr_data.create_dataset('left_robot_tcp_pos', data=robot0_eef_pos, chunks=(1000, 3), dtype='float32', compressor=compressor)
# zarr_data.create_dataset('left_robot_tcp_pose', data=robot0_eef_pos, chunks=(1000, 3), dtype='float32', compressor=compressor)

# zarr_data.create_dataset('action', data=robot0_absolute_action_4d, chunks=(1000, 4), dtype='float32', compressor=compressor)
# zarr_data.create_dataset('target', data=robot0_absolute_action_4d, chunks=(1000, 4), dtype='float32', compressor=compressor)

# === [IMPORTANT] action + target ===


zarr_data.create_dataset('robot0_demo_start_pose', data=robot0_demo_start_pose, chunks=(1000, 6), dtype='float64', compressor=compressor)
zarr_data.create_dataset('robot0_demo_end_pose', data=robot0_demo_end_pose, chunks=(1000, 6), dtype='float64', compressor=compressor)

zarr_meta.create_dataset('episode_ends', data=episode_ends, chunks=(1000,), dtype='int64')

print("✅ Done saving ReplayBuffer zarr format!")
print(zarr_data.tree())
