import hydra
import h5py
import cv2
import numpy as np
from loguru import logger
from omegaconf import DictConfig
import os
from image_util import draw_coordinate_frame


@hydra.main(version_base=None, config_path="config", config_name="convert_dataset")
def main(cfg: DictConfig) -> None:
    # Get list of processed scene files
    scene_files = sorted([f for f in os.listdir(cfg.dataset_target_directory) 
                         if f.startswith('scene_') and f.endswith('.hdf5')])
    
    for scene_file in scene_files:
        file_path = os.path.join(cfg.dataset_target_directory, scene_file)
        logger.info(f"Loading processed scene from {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            heightmap = f['heightmap'][()]
            colormap = f['colormap'][()]
            workspace_bounds = f['workspace_bounds'][()]
            grasp_pose = f['grasp_pose'][()]
            
            # Normalize heightmap for visualization
            valid_mask = heightmap != -np.inf
            if valid_mask.any():
                normalized_heightmap = np.zeros_like(heightmap)
                normalized_heightmap[valid_mask] = (heightmap[valid_mask] - workspace_bounds[2][0]) / (workspace_bounds[2][1] - workspace_bounds[2][0])
            else:
                normalized_heightmap = np.zeros_like(heightmap)
            
            # Create a copy of colormap for visualization
            vis_colormap = colormap.copy()
            
            # Draw coordinate frame on colormap
            vis_colormap = draw_coordinate_frame(
                vis_colormap, 
                grasp_pose, 
                workspace_bounds,
                cfg.projection_resolution
            )
            
            # Display
            cv2.imshow('Heightmap', normalized_heightmap)
            cv2.imshow('Colormap with Grasp Pose', vis_colormap)
            
            key = cv2.waitKey(0)
            if key == ord('q'):  # Press 'q' to quit
                break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
