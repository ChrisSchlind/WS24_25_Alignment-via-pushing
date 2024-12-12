import h5py
import os
import numpy as np
from loguru import logger


def store_orthographic_data(scene_id, heightmap, colormap, grasp_pose, workspace_bounds, image_stats, target_directory):
    """Stores processed data in HDF5 format."""
    os.makedirs(target_directory, exist_ok=True)
    file_name = os.path.join(target_directory, f'scene_{scene_id:04d}.hdf5')
    
    logger.info(f"Storing processed data to {file_name}")
    with h5py.File(file_name, 'w') as f:
        # Store heightmap and colormap with compression
        f.create_dataset('heightmap', 
                        data=heightmap,
                        compression='gzip')
        
        f.create_dataset('colormap', 
                        data=colormap,
                        chunks=(colormap.shape[0], colormap.shape[1], 3),
                        compression='gzip')
        
        # Store grasp pose and workspace bounds
        f.create_dataset('grasp_pose', data=grasp_pose)
        f.create_dataset('workspace_bounds', data=np.array(workspace_bounds))
        
        # Create image_stats group and store as individual datasets
        stats_group = f.create_group('image_stats')
        for key, value in image_stats.items():
            stats_group.create_dataset(key, data=np.array(value))


def load_image_stats(file_name):
    """Load image statistics from HDF5 file."""
    with h5py.File(file_name, 'r') as f:
        stats = {}
        for key in f['image_stats'].keys():
            stats[key] = f['image_stats'][key][()]
    return stats