# %%
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
import matplotlib.pyplot as plt

from umi.common.pose_util import pose_to_mat, mat_to_pose10d
from scipy.spatial.transform import Rotation as R
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

from loguru import logger

@click.command()
@click.option('-o', '--input', required=True, help='Zarr path')
def main(input):
    # === [0] 입력 경로와 출력 경로 설정 ===
    dataset_path = input  # ex) ~/xxx/dataset.zarr.zip
    mean_std_dir = os.path.dirname(dataset_path)
    output_path = dataset_path.replace(".zip", "_normalized_temp")   # zip → 디렉토리로 열기
    store = zarr.DirectoryStore(output_path)

    # === [1] 기존 zip zarr을 DirectoryStore로 복사 (ReplayBuffer 사용) ===
    replay_buffer = ReplayBuffer.create_from_path(dataset_path)
    replay_buffer.save_to_store(store)

    # === [2] zarr 열기 ===
    dataset = zarr.open(store, mode="a")   # 저장 가능한 모드로 열기

    # === [3] tactile offset 불러오기 /xela ===
    marker_offset = dataset['data/camera0_tactile_offset']
    print("marker_offset shape:", marker_offset.shape)

    N = marker_offset.shape[0]
    marker_offset_flat = marker_offset[:]
    print("marker_offset_flat shape:", marker_offset_flat.shape)


    # === [4] normalize ===
    mean = marker_offset_flat.mean(axis=0)
    std = marker_offset_flat.std(axis=0) + 1e-8
    print(f"mean: {mean}")
    print(f"std: {std}")
    normalized_offset = ((marker_offset_flat - mean) / std).astype(np.float32)
    np.save(os.path.join(mean_std_dir, "tactile_normalization_mean.npy"), mean)
    np.save(os.path.join(mean_std_dir, "tactile_normalization_std.npy"), std)
    # === [5] 저장 ===
    if "data/camera0_tactile_normalized" in dataset:
        del dataset["data/camera0_tactile_normalized"]

    dataset.create_dataset("data/camera0_tactile_normalized", data=normalized_offset, chunks=(1024, 16), dtype=np.float32)

    print("Done saving normalized tactile offset!")

    # === [6] absolute action ===
    eef_pos = dataset["data/robot0_eef_pos"][:]           
    eef_rot = dataset["data/robot0_eef_rot_axis_angle"][:]  
    gripper_width = dataset["data/robot0_gripper_width"][:]  
    demo_end_pose = dataset["data/robot0_demo_end_pose"][:]  
    episode_ends = dataset["meta/episode_ends"][:]

    N = len(eef_pos)

    # Convert to 9D pose using pose_util(UMI)
    eef_pose6d = np.concatenate([eef_pos, eef_rot], axis=1)
    print("eef_pose6d: ", eef_pose6d) 
    eef_pose_mat = pose_to_mat(eef_pose6d)  # umi code use 
    print("eef_pose_mat umi:", eef_pose_mat)
    eef_pose_9d = mat_to_pose10d(eef_pose_mat)
    print("eef_pose_9d umi:", eef_pose_9d)

    # Verify against manual conversion
    eef_rot_mat = R.from_rotvec(eef_rot).as_matrix()
    eef_rot_6d_manual = eef_rot_mat[...,:2].reshape(N, 6)
    eef_pose_9d_manual = np.concatenate([eef_pos, eef_rot_6d_manual], axis=1)

    absolute_action = np.zeros((N, 10), dtype=np.float32)
    absolute_action_4d = np.zeros((N, 4), dtype=np.float32)
    
    if np.allclose(eef_pose_9d, eef_pose_9d_manual):
        print('9D conversion verification passed.')
    else:
        print('Warning: 9D conversion mismatch!')


    start_idx = 0
    for end_idx in tqdm(episode_ends, desc="Generating absolute action"):
        logger.info(f"end_idx = {end_idx}")
        for i in range(start_idx, end_idx):
            if i + 1 < end_idx:
                # logger.debug(f"index = {i}")
                next_pose = eef_pose_9d[i + 1]
                next_gripper = gripper_width[i + 1]
                absolute_action[i] = np.concatenate([next_pose, next_gripper.flatten()])

                next_pos = eef_pos[i+1]
                absolute_action_4d[i] = np.concatenate([next_pos, next_gripper.flatten()])
                
            else:
                # logger.debug(f"end_index = {i}")
                end_pose = demo_end_pose[i]
                end_pos = end_pose[:3]
                end_rot = end_pose[3:]
                end_rot_mat = R.from_rotvec(end_rot).as_matrix()
                end_rot_6d_manual = end_rot_mat[:,:2].reshape(6) ### manual

                # use pose_util
                end_pose6d = np.concatenate([end_pos, end_rot])
                end_pose_mat = pose_to_mat(end_pose6d)
                end_pose_9d = mat_to_pose10d(end_pose_mat)

                # absolute_action[i] = np.concatenate([end_pos, end_rot_6d_manual, gripper_width[i].flatten()])
                # logger.debug(f"absolute_action 1 = {absolute_action[i-1]}")
                absolute_action[i] = np.concatenate([end_pose_9d, gripper_width[i].flatten()])
                absolute_action_4d[i] = np.concatenate([end_pos, gripper_width[i].flatten()])
                # logger.debug(f"absolute_action 2 = {absolute_action[i]}")
        start_idx = end_idx

    # === [7] 저장 ===
    for key in ["data/robot0_absolute_action", "data/robot0_absolute_action_4d", "data/robot0_eef_pose_9d"]:
        if key in dataset:
            del dataset[key]

    dataset.create_dataset("data/robot0_absolute_action", data=absolute_action, chunks=(1024, 10), dtype=np.float32)
    dataset.create_dataset("data/robot0_absolute_action_4d", data=absolute_action_4d, chunks=(1024, 4), dtype=np.float32)
    dataset.create_dataset("data/robot0_eef_pose_9d", data=eef_pose_9d, chunks=(1024, 9), dtype=np.float32) # (pos + rot6d) UMI setup

    # === [8] DirectoryStore -> 다시 zip으로 압축 ===
    zip_output_path = dataset_path.replace(".zip", "_normalized.zarr.zip")
    shutil.make_archive(zip_output_path.replace('.zip', ''), 'zip', output_path)

    print(f"✅ Final zipped dataset saved to: {zip_output_path}")

    # === [9] tree 확인 ===
    print(dataset.tree())

if __name__ == "__main__":
    main()
