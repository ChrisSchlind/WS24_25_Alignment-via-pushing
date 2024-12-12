import sys
import hydra
from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate
import cv2
import h5py
import os
import pickle
from bullet_env.util import setup_bullet_client, stdout_redirected
from image_util import draw_pose
from transform.affine import Affine

@hydra.main(version_base=None, config_path="config", config_name="tn_train_data")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # Setup bullet client for visualization
    bullet_client = setup_bullet_client(render=True)

    env = instantiate(cfg.env, bullet_client=bullet_client)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)

    # Get list of all scene files in the dataset directory
    scene_files = sorted([f for f in os.listdir(cfg.dataset_directory) 
                         if f.startswith('scene_') and f.endswith('.hdf5')])

    for scene_file in scene_files:
        file_path = os.path.join(cfg.dataset_directory, scene_file)
        logger.info(f"Loading scene from {file_path}")

        robot.home()
        robot.gripper.open()
        with h5py.File(file_path, 'r') as f:
            # Load task info
            task_info = pickle.loads(f['task_info'][()])
            task = instantiate(task_info)
            task.setup(env)
            # Load grasp pose
            grasp_pose = f['grasp_pose'][()]

            # Load and display observations
            obs_group = f['observations']
            for obs_name in obs_group.keys():
                obs = obs_group[obs_name]
                
                # Get RGB image and camera parameters
                rgb_image = obs['rgb'][()]
                extrinsics = obs['extrinsics'][()]
                intrinsics = obs['intrinsics'][()]

                # Create visualization
                image_copy = rgb_image.copy()
                draw_pose(extrinsics, grasp_pose, intrinsics, image_copy)
                
                # Display images
                cv2.imshow('RGB with Pose', image_copy)
                
                depth_image = obs['depth'][()]
                depth_image = depth_image / 2.0
                cv2.imshow('Depth', depth_image)
                
                # Wait for key press
                key_pressed = cv2.waitKey(0)
                if key_pressed == ord('q'):
                    return
            env.spawn_coordinate_frame(grasp_pose)
            action = Affine.from_matrix(grasp_pose)
            pre_grasp_offset = Affine([0, 0, -0.1])
            pre_grasp_pose = action * pre_grasp_offset
            robot.ptp(pre_grasp_pose)
            robot.lin(action)
            robot.gripper.close()
            robot.lin(pre_grasp_pose)
            env.remove_coordinate_frames()
            task.clean(env)

    with stdout_redirected():
        bullet_client.disconnect()

if __name__ == "__main__":
    main()
