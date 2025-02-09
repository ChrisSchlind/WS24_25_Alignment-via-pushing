import sys
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy
import cv2
from bullet_env.util import setup_bullet_client, stdout_redirected
from util.convert_util import convert_to_orthographic, display_orthographic

@hydra.main(version_base=None, config_path="config", config_name="test_env")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # Setup bullet client
    bullet_client = setup_bullet_client(cfg.render)
    logger.info("Bullet client setup completed.")

    # Instantiate UR10Cell
    robot = instantiate(cfg.robot, bullet_client=bullet_client)
    logger.info("Robot instantiation completed.")

    # Create task factory and other components first
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    t_bounds[2, 1] = t_bounds[2, 0]
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    t_center = np.mean(t_bounds, axis=1)
    logger.info("Task factory instantiation completed.")

    # Instantiate teletentric camera
    teletentric_camera = instantiate(cfg.teletentric_camera, bullet_client=bullet_client, t_center=t_center, robot=robot)
    logger.info("Teletentric camera instantiation completed.")

    # Instantiate camera factory
    camera_factory = instantiate(cfg.camera_factory, bullet_client=bullet_client, t_center=t_center)
    logger.info("Camera factory instantiation completed.")

    # Create environment with all components
    pushing_env = instantiate(cfg.pushing_env, bullet_client=bullet_client, robot=robot, task_factory=task_factory, teletentric_camera=teletentric_camera)
    logger.info("Environment instantiation completed.")

    if(cfg.debug): logger.info("Instantiation completed.")

    for _ in range(10):
        robot.home()
        task = task_factory.create_task()
        task.setup(pushing_env)
        observations = [camera.get_observation() for camera in camera_factory.cameras]

        if cfg.debug:
            # 1. Standard cameras
            image_copy = copy.deepcopy(observations[0]["rgb"])
            # Convert to rgb for visualization
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            cv2.imshow("Normal Camera No. 0: RGB", image_copy)
            depth_copy = copy.deepcopy(observations[0]["depth"])
            # rescale for visualization
            depth_copy = depth_copy / 2.0
            cv2.imshow("Normal Camera No. 0: Depth", depth_copy)

            # Convert observations to orthographic view
            height_map, colormap = convert_to_orthographic(observations, cfg.workspace_bounds, cfg.projection_resolution)
            # Display orthographic view
            display_orthographic(height_map, colormap, cfg.workspace_bounds)

            # 2. Teletentric camera
            teletentric_observation, _ = teletentric_camera.get_observation()

            teletentric_image_copy = copy.deepcopy(teletentric_observation["rgb"])
            # Convert to rgb for visualization
            teletentric_image_copy = cv2.cvtColor(teletentric_image_copy, cv2.COLOR_BGR2RGB)
            cv2.imshow("Teletentric Camera: RGB", teletentric_image_copy)
            teletentric_depth_copy = copy.deepcopy(teletentric_observation["depth"])
            # rescale for visualization
            teletentric_depth_copy = teletentric_depth_copy / 2.0
            cv2.imshow("Teletentric Camera: Depth", teletentric_depth_copy)

            pressed_key = cv2.waitKey(0)
            if pressed_key == ord("q"):
                break

        task.clean(pushing_env)

    pushing_env.close()

    with stdout_redirected():
        bullet_client.disconnect()

if __name__ == "__main__":
    main()
