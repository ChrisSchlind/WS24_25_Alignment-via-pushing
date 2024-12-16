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
    t_center = np.mean(t_bounds, axis=1)
    camera_factory = instantiate(cfg.camera_factory, bullet_client=bullet_client, t_center=t_center)

    logger.info("Instantiation completed.")

    
    robot.home()
    task = task_factory.create_task()
    task.setup(env)
    observations = [camera.get_observation() for camera in camera_factory.cameras]


    obj, area = task.get_object_and_area_with_same_id(0)
    print("Object: ", obj)
    print("Area: ", area)

    

    
    
    cv2.waitKey(0)

    task.clean(env)

    with stdout_redirected():
        bullet_client.disconnect()


if __name__ == "__main__":
    main()