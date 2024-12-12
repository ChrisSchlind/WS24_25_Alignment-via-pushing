import sys
import hydra
from loguru import logger
from omegaconf import DictConfig
import cv2
import h5py
import os
import numpy as np
from data_util import store_orthographic_data

def create_pointcloud(rgb, depth, intrinsics):
    """Creates a pointcloud from RGB-D image and camera intrinsics."""
    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel coordinates grid
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    # Convert to 3D points
    z = depth
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    
    # Stack coordinates and reshape
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    
    # Remove invalid points (zero depth)
    mask = z.reshape(-1) > 0
    return points[mask], colors[mask]

def transform_pointcloud(points, extrinsics):
    """Transforms pointcloud to world coordinates."""
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    return (points @ R.T) + t

def filter_workspace(points, colors, bounds):
    """Filters points within workspace bounds.
    
    Args:
        points: Nx3 array of points
        colors: Nx3 array of RGB colors
        bounds: List of [min, max] pairs for x, y, z dimensions
    """
    x_bounds, y_bounds, z_bounds = bounds
    mask = ((points[:, 0] >= x_bounds[0]) & (points[:, 0] <= x_bounds[1]) &
           (points[:, 1] >= y_bounds[0]) & (points[:, 1] <= y_bounds[1]) &
           (points[:, 2] >= z_bounds[0]) & (points[:, 2] <= z_bounds[1]))
    return points[mask], colors[mask]

def create_heightmap(points, colors, resolution, bounds):
    """Projects points to 2D heightmap with colors and heights.
    
    Args:
        points: Nx3 array of points
        colors: Nx3 array of RGB colors
        resolution: [height, width] in pixels
        bounds: List of [min, max] pairs for x, y, z dimensions
    """
    x_bounds, y_bounds, z_bounds = bounds
    height, width = resolution
    
    # Calculate meter per pixel
    x_scale = (x_bounds[1] - x_bounds[0]) / width
    y_scale = (y_bounds[1] - y_bounds[0]) / height
    
    # Initialize heightmap arrays
    heightmap = np.full((height, width), z_bounds[0])
    colormap = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert points to grid coordinates
    x_coords = ((points[:, 0] - x_bounds[0]) / x_scale).astype(int)
    y_coords = ((points[:, 1] - y_bounds[0]) / y_scale).astype(int)
    
    # Valid grid coordinates
    mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    
    # Update heightmap and colormap
    for i in range(len(points)):
        if mask[i]:
            x, y = x_coords[i], y_coords[i]
            z = points[i, 2]
            if z > heightmap[y, x]:
                heightmap[y, x] = z
                colormap[y, x] = colors[i]
    return heightmap, colormap

@hydra.main(version_base=None, config_path="config", config_name="convert_dataset")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # Get list of all scene files in the dataset directory
    scene_files = sorted([f for f in os.listdir(cfg.dataset_source_directory) 
                         if f.startswith('scene_') and f.endswith('.hdf5')])
    
    height_maps = []
    colormaps = []
    grasp_poses = []

    for scene_file in scene_files:
        file_path = os.path.join(cfg.dataset_source_directory, scene_file)
        logger.info(f"Loading scene from {file_path}")            
        
        with h5py.File(file_path, 'r') as f:
            grasp_pose = f['grasp_pose'][()]
            # Load and process observations
            obs_group = f['observations']
            all_points = []
            all_colors = []
            
            for obs_name in obs_group.keys():
                obs = obs_group[obs_name]
                rgb_image = obs['rgb'][()]
                extrinsics = obs['extrinsics'][()]
                intrinsics = obs['intrinsics'][()]
                depth_image = obs['depth'][()]
                
                # Create and transform pointcloud
                points, colors = create_pointcloud(rgb_image, depth_image, intrinsics)
                points = transform_pointcloud(points, extrinsics)
                all_points.append(points)
                all_colors.append(colors)
            
            # Combine all pointclouds
            points = np.concatenate(all_points, axis=0)
            colors = np.concatenate(all_colors, axis=0)
            
            # Filter workspace
            points, colors = filter_workspace(points, colors, cfg.workspace_bounds)
            
            # Create heightmap
            heightmap, colormap = create_heightmap(points, colors, 
                                                   cfg.projection_resolution, 
                                                   cfg.workspace_bounds)
            if cfg.debug:
                scaled_heightmap = (heightmap - cfg.workspace_bounds[2][0]) / (cfg.workspace_bounds[2][1] - cfg.workspace_bounds[2][0])
                cv2.imshow('Heightmap', scaled_heightmap)
                cv2.imshow('Colormap', colormap)
                cv2.waitKey(0)
            
            height_maps.append(heightmap)
            colormaps.append(colormap)
            grasp_poses.append(grasp_pose)
    
    if cfg.store_dataset:   
        # compute mean and std of all channels
        mean_heightmap = np.mean(height_maps)
        std_heightmap = np.std(height_maps)
        mean_colormap = np.mean(colormaps)
        std_colormap = np.std(colormaps)
        image_stats = {
            'mean_heightmap': mean_heightmap,
            'std_heightmap': std_heightmap,
            'mean_colormap': mean_colormap,
            'std_colormap': std_colormap
        }
        for i, (heightmap, colormap, grasp_pose) in enumerate(zip(height_maps, colormaps, grasp_poses)):
            standardized_heightmap = (heightmap - mean_heightmap) / std_heightmap
            standardized_colormap = (colormap - mean_colormap) / std_colormap
            store_orthographic_data(i, standardized_heightmap, standardized_colormap, grasp_pose, cfg.workspace_bounds, image_stats, cfg.dataset_target_directory)

if __name__ == "__main__":
    main()
    