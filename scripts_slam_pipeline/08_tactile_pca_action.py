# %%
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

save_dir = "/home/embodied-ai/mcy/reactive_diffusion_policy/data/PCA_Transform_DIGIT"

@click.command()
@click.option('-o', '--input', required=True, help='Zarr path')


def main(input):
    # === [0] 입력 경로와 출력 경로 설정 ===
    dataset_path = input  # ex) ~/xxx/dataset.zarr.zip
    output_path = dataset_path.replace(".zip", "_pca_temp")   # zip → 디렉토리로 열기
    store = zarr.DirectoryStore(output_path)

    # === [1] 기존 zip zarr을 DirectoryStore로 복사 (ReplayBuffer 사용) ===
    replay_buffer = ReplayBuffer.create_from_path(dataset_path)
    replay_buffer.save_to_store(store)

    # === [2] zarr 열기 ===
    dataset = zarr.open(store, mode="a")   # 저장 가능한 모드로 열기

    # === [3] tactile offset 불러오기 ===
    marker_offset = dataset['data/camera0_tactile_offset']
    print("marker_offset shape:", marker_offset.shape)

    N = marker_offset.shape[0]
    marker_offset_flat = marker_offset[:, :, 0]
    print("marker_offset_flat shape:", marker_offset_flat.shape)

    # === [4] zero-only 데이터 제외 ===
    non_zero_idx = (marker_offset_flat != 0).any(axis=1)
    marker_offset_flat_valid = marker_offset_flat[non_zero_idx]

    if marker_offset_flat_valid.shape[0] < 10:
        print("❌ Non-zero 데이터가 너무 적어 PCA 학습을 건너뜁니다.")
        return

    # === [5] PCA 학습 ===
    n_components = 15
    pca = PCA(n_components=n_components)

    print("PCA fitting...")
    pca_emb_valid = pca.fit_transform(marker_offset_flat_valid)
    
    pca_emb_all = np.zeros((N, n_components), dtype=np.float32)
    if non_zero_idx.any():
        pca_emb_all[non_zero_idx] = pca.transform(marker_offset_flat_valid)

    print("PCA embedding shape:", pca_emb_all.shape)
    print("pca explain variance", pca.explained_variance_ratio_)
    print("pca explain variance sum", np.sum(pca.explained_variance_ratio_))

    # -- [5-1] absolute action --
    eef_pos = dataset["data/robot0_eef_pos"][:]           
    eef_rot = dataset["data/robot0_eef_rot_axis_angle"][:]  
    gripper_width = dataset["data/robot0_gripper_width"][:]  
    demo_end_pose = dataset["data/robot0_demo_end_pose"][:]  
    episode_ends = dataset["meta/episode_ends"][:]

    N = len(eef_pos)
    eef_rot_mat = R.from_rotvec(eef_rot).as_matrix()
    eef_rot_6d = eef_rot_mat[...,:2].reshape(N, 6)

    absolute_action = np.zeros((N, 10), dtype=np.float32)
    absolute_action_4d = np.zeros((N, 4), dtype=np.float32)
    eef_pose_9d = np.concatenate([eef_pos, eef_rot_6d], axis=1)

    start_idx = 0
    for end_idx in tqdm(episode_ends, desc="Generating absolute action"):
        for i in range(start_idx, end_idx):
            if i + 1 < end_idx:
                next_pose = eef_pose_9d[i + 1]
                next_gripper = gripper_width[i + 1]
                absolute_action[i] = np.concatenate([next_pose, next_gripper.flatten()])

                next_pos = eef_pos[i+1]
                absolute_action_4d[i] = np.concatenate([next_pos, next_gripper.flatten()])
                
            else:
                end_pose = demo_end_pose[i]
                end_pos = end_pose[:3]
                end_rot = end_pose[3:]
                end_rot_mat = R.from_rotvec(end_rot).as_matrix()
                end_rot_6d = end_rot_mat[:,:2].reshape(6)

                absolute_action[i] = np.concatenate([end_pos, end_rot_6d, gripper_width[i].flatten()])
                absolute_action_4d[i] = np.concatenate([end_pos, gripper_width[i].flatten()])
        start_idx = end_idx


    # === [6] 저장 ===
    if "data/camera0_pca_emb" in dataset:
        del dataset["data/camera0_pca_emb"]

    if "data/robot0_absolute_action" in dataset:
        del dataset["data/robot0_absolute_action"]
    
    if "data/robot0_absolute_action_4d" in dataset:
        del dataset["data/robot0_absolute_action_4d"]
    
    if "data/robot0_eef_pose_9d" in dataset:
        del dataset["data/robot0_eef_pose_9d"]

    dataset.create_dataset("data/camera0_pca_emb", data=pca_emb_all, chunks=(1024, 15), dtype=np.float32)
    dataset.create_dataset("data/robot0_absolute_action", data=absolute_action, chunks=(1024, 10), dtype=np.float32)
    dataset.create_dataset("data/robot0_absolute_action_4d", data=absolute_action_4d, chunks=(1024, 4), dtype=np.float32)
    dataset.create_dataset("data/robot0_eef_pose_9d", data=eef_pose_9d, chunks=(1024, 9), dtype=np.float32)
    
    # PCA transform matrix, mean 저장
    os.makedirs(save_dir, exist_ok=True)
    transform_path = os.path.join(save_dir, "pca_transform_matrix.npy")
    mean_path = os.path.join(save_dir, "pca_mean_matrix.npy")

    np.save(transform_path, pca.components_.T)  # (63, 15)
    np.save(mean_path, pca.mean_)               # (63,)

    print("Done saving PCA embedding and transform matrices!")

    # === [7] DirectoryStore -> 다시 zip으로 압축 ===
    zip_output_path = dataset_path.replace(".zip", "_pca.zarr.zip")
    shutil.make_archive(zip_output_path.replace('.zip', ''), 'zip', output_path)

    print(f"✅ Final zipped dataset saved to: {zip_output_path}")

    # === [8] tree 확인 ===
    print(dataset.tree())

if __name__ == "__main__":
    main()
