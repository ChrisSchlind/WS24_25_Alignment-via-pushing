import h5py
import numpy as np
import os
import pickle
from loguru import logger


def store_data_grasp(scene_id, task_info, observations, grasp_pose, dataset_directory=None):
    if dataset_directory is None:
        file_name = f'scene_{scene_id:04d}.hdf5'
    else:
        os.makedirs(dataset_directory, exist_ok=True)
        file_name = os.path.join(dataset_directory, f'scene_{scene_id:04d}.hdf5')
    logger.info(f"Storing data to {file_name}")
    with h5py.File(file_name, 'w') as f:
        # Store task_info as a JSON-like string or individual datasets
        task_info_serialized = pickle.dumps(task_info)
        f.create_dataset("task_info", data=np.frombuffer(task_info_serialized, dtype='uint8'))
        # Create a group for observations
        obs_group = f.create_group('observations')
        for i, obs in enumerate(observations):
            obs_id = f'observation_{i:04d}'
            obs_subgroup = obs_group.create_group(obs_id)
            # Define chunk size for the RGB image dataset (e.g., one full image per chunk)
            chunk_size = (480, 640, 3)
            # Store RGB image with chunking and compression
            obs_subgroup.create_dataset(
                'rgb',
                data=obs['rgb'][..., :3],
                chunks=chunk_size,  # Define chunk size
                compression='gzip',  # Use compression
            )
            depth_chunk_size = (480, 640)
            obs_subgroup.create_dataset(
                'depth',
                data=obs['depth'],
                chunks=depth_chunk_size,  # Define chunk size
                compression='gzip',  # Use compression
            )
            # Define chunk sizes for camera matrices (since they are small, use full size as chunks)
            extrinsics_chunk_size = (4, 4)
            intrinsics_chunk_size = (3, 3)
            # Store camera extrinsic matrix
            obs_subgroup.create_dataset(
                'extrinsics',
                data=obs['extrinsics'],
                chunks=extrinsics_chunk_size
            )
            # Store camera intrinsic matrix
            obs_subgroup.create_dataset(
                'intrinsics',
                data=obs['intrinsics'],
                chunks=intrinsics_chunk_size
            )
        f.create_dataset('grasp_pose', data=grasp_pose)
