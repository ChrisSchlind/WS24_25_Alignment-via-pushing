import sys
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy
import cv2

from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine
from convert_util import convert_to_orthographic, display_orthographic

from image_util import draw_pose

@hydra.main(version_base=None, config_path="config", config_name="test_env")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    bullet_client = setup_bullet_client(cfg.render)

    env = instantiate(cfg.env, bullet_client=bullet_client)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    # the bounds for objects should be on the ground plane of the robots workspace
    t_bounds[2, 1] = t_bounds[2, 0]
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    #oracle = instantiate(cfg.oracle)
    t_center = np.mean(t_bounds, axis=1)
    camera_factory = instantiate(cfg.camera_factory, bullet_client=bullet_client, t_center=t_center)

    logger.info("Instantiation completed.")

    for _ in range(10):
        robot.home()
        task = task_factory.create_task()
        task.setup(env)
        observations = [camera.get_observation() for camera in camera_factory.cameras]

        if cfg.debug:
            image_copy = copy.deepcopy(observations[0]["rgb"])
            cv2.imshow("rgb", image_copy)
            depth_copy = copy.deepcopy(observations[0]["depth"])
            # rescale for visualization
            depth_copy = depth_copy / 2.0
            cv2.imshow("depth", depth_copy)

            # Convert observations to orthographic view
            height_map, colormap = convert_to_orthographic(observations, cfg.workspace_bounds, cfg.projection_resolution)
            # Display orthographic view
            display_orthographic(height_map, colormap, cfg.workspace_bounds)

            pressed_key = cv2.waitKey(0)
            if pressed_key == ord('q'):
                break

        task.clean(env)

    with stdout_redirected():
        bullet_client.disconnect()


if __name__ == "__main__":
    main()
