import numpy as np
import cv2


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

def convert_to_orthographic(observations, workspace_bounds, projection_resolution):
    """Converts observations to orthographic heightmap and colormap."""

    all_points = []
    all_colors = []
    
    for i in range(len(observations)):
        obs = observations[i]
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
    points, colors = filter_workspace(points, colors, workspace_bounds)
    
    # Create heightmap
    heightmap, colormap = create_heightmap(points, colors, 
                                            projection_resolution, 
                                            workspace_bounds)

    return heightmap, colormap

def display_orthographic(heightmap, colormap, workspace_bounds):
    """Displays orthographic heightmap and colormap."""

    # Normalize heightmap for visualization
    valid_mask = heightmap != -np.inf
    if valid_mask.any():
        normalized_heightmap = np.zeros_like(heightmap)
        normalized_heightmap[valid_mask] = (heightmap[valid_mask] - workspace_bounds[2][0]) / (workspace_bounds[2][1] - workspace_bounds[2][0])
    else:
        normalized_heightmap = np.zeros_like(heightmap)

    cv2.imshow('Heightmap', heightmap)
    cv2.imshow('Colormap', colormap)
    cv2.waitKey(0)            