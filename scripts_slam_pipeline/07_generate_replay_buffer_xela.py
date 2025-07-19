# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import re
from datetime import datetime, timedelta
# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()
import torch
from digit_depth.digit.digit_sensor import DigitSensor
from digit_depth.train.prepost_mlp import *
from digit_depth.handlers import find_recent_model
from digit_depth.third_party import geom_utils
from pathlib import Path
import yaml

# %%
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
@click.option('-ms', '--mirror_swap', is_flag=True, default=False)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, out_res, out_fov, compression_level, 
         no_mirror, mirror_swap, num_workers):
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
        
    out_res = tuple(int(x) for x in out_res.split(','))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )
        
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # dump lowdim data to replay buffer
    # generate argumnet for videos
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))
        
        videos_dict = defaultdict(list)
        for plan_episode in plan:
            grippers = plan_episode['grippers']
            
            # check that all episodes have the same number of grippers 
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)
                
            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)
                
            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):    
                eef_pose = gripper['tcp_pose']
                eef_pos = eef_pose[...,:3]
                eef_rot = eef_pose[...,3:]
                gripper_widths = gripper['gripper_width']
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']
                
                robot_name = f'robot{gripper_id}'
                episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
                episode_data[robot_name + '_demo_end_pose'] = demo_end_pose
            
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            # aggregate video gen aguments
            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()
                
                video_start, video_end = camera['video_start_end']
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)
                
                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })
            buffer_start += n_frames
        
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")
    
    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
    
    # dump images + tactile offset
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

        tactile_offset_name = f'camera{cam_id}_tactile_offset'
        _ = out_replay_buffer.data.require_dataset(
            name=tactile_offset_name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0], 16),
            chunks=(1, 16),
            compressor=None,  # 또는 float 지원 가능한 압축
            dtype=np.float32 
        )

    def extract_timestamp_from_path(path_str):
        match = re.search(r"(\d{4})[.-](\d{2})[.-](\d{2})[_\.](\d{2})[.-](\d{2})[.-](\d{2})[._](\d+)", path_str)
        if match:
            year, month, day, hour, minute, second, microsec = match.groups()
            timestamp_str = f"{year}-{month}-{day} {hour}:{minute}:{second}.{microsec[:6]:0<6}"
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        else:
            raise ValueError(f"\u274c No timestamp found in {path_str}")

    def video_to_zarr_from_csv(replay_buffer, mp4_path, tasks, out_res, camera_idx):
        
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        curr_task_idx = 0
        
        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh,ow,3),dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
            is_mirror = (mirror_mask[...,0] == 0)

        synched_csv_path = os.path.join(os.path.dirname(mp4_path), 'synched_tactile.csv')
        df_tactile = pd.read_csv(synched_csv_path)
        if 'time' not in df_tactile.columns or df_tactile.shape[1] != 17:
            print(f"[SKIP] Invalid tactile csv: {synched_csv_path}")
            return

        # CSV absolute time 
        # tactile_times = df_tactile['time'].values
        # CSV 상대 시간 (frame 기준, 100 fps 가정)
        tactile_times = np.arange(len(df_tactile)) / 100.0
        tactile_values = df_tactile.drop(columns='time').values

        tactile_offset_name = f'camera{camera_idx}_tactile_offset'
        tactile_offset_array = replay_buffer.data[tactile_offset_name]

        # video_start_time = extract_timestamp_from_path(str(mp4_path))

        cap = cv2.VideoCapture(mp4_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        overlay_writer = None
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            in_stream.thread_count = 1
            # fps = in_stream.average_rate if in_stream.average_rate is not None else 30
            fps = float(fps)

            curr_task_idx = 0
            buffer_idx = 0
            matched_taxels_all = []
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    break

                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']

                    # do current task
                    img = frame.to_ndarray(format='rgb24')

                    # inpaint tags
                    this_det = tag_detection_results[frame_idx]
                    all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                    for corners in all_corners:
                        img = inpaint_tag(img, corners)
                        
                    # mask out gripper
                    img = draw_predefined_mask(img, color=(0,0,0), 
                        mirror=no_mirror, gripper=True, finger=False)
                    # resize
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)

                    # xela matching 
                    rel_time_sec = frame_idx/fps
                    # t_sec = frame_time.timestamp()
                    # idx = np.argmin(np.abs(tactile_times - t_sec))
                    idx = np.argmin(np.abs(tactile_times - rel_time_sec))
                    matched_taxel = tactile_values[idx].astype(np.float32)

                    # handle mirror swap
                    if mirror_swap:
                        img[is_mirror] = img[:,::-1,:][is_mirror]
                        
                    # compress image
                    img_array[buffer_idx] = img

                    tactile_offset_array[buffer_idx] = matched_taxel

                    buffer_idx += 1

                    # === 🆕 overlay 및 저장 ===
                    if frame_idx % 10 == 0:
                        # 텍스트로 overlay
                        overlay_img = img.copy()
                        txt = ','.join(f"{v:.0f}" for v in matched_taxel[-4:])  # 마지막 4개 taxel만 표시 (간결)
                        cv2.putText(
                            overlay_img,
                            f"Taxel: {txt}",
                            (10, out_res[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 0, 0),
                            1,
                            lineType=cv2.LINE_AA
                        )

                        if overlay_writer is None:
                            overlay_path = os.path.join(os.path.dirname(mp4_path), f"overlay_preview_camera{camera_idx}.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            overlay_writer = cv2.VideoWriter(overlay_path, fourcc, 6, out_res)  # 6fps for 10-frame step
                            print(f"🎥 Overlay 저장: {overlay_path}")

                        overlay_writer.write(cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

                        matched_taxels_all.append(matched_taxel)
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        curr_task_idx += 1
                else:
                    assert False
            matched_taxels_all = np.array(matched_taxels_all)
            csv_save_path = os.path.join(os.path.dirname(mp4_path), f"matched_taxels_camera{camera_idx}.csv")
            np.savetxt(csv_save_path, matched_taxels_all, delimiter=",", fmt="%.5f")
            print(f"✅ Matched taxels saved to {csv_save_path}")

        if overlay_writer is not None:
            overlay_writer.release()

    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                camera_idx = tasks[0]['camera_idx']
                futures.add(executor.submit(video_to_zarr_from_csv, 
                    out_replay_buffer, mp4_path, tasks, out_res, camera_idx))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")



if __name__ == "__main__":
    main()
