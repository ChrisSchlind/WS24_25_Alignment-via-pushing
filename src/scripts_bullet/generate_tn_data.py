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

from image_util import draw_pose
from data_util import store_data_grasp


@hydra.main(version_base=None, config_path="config", config_name="tn_train_data")
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
    oracle = instantiate(cfg.oracle)
    t_center = np.mean(t_bounds, axis=1)
    camera_factory = instantiate(cfg.camera_factory, bullet_client=bullet_client, t_center=t_center)

    for i in range(cfg.n_scenes):
        robot.home()
        robot.gripper.open()
        task = task_factory.create_task()
        task.setup(env)
        task_info = task.get_info()
        pose = oracle.solve(task)
        observations = [camera.get_observation() for camera in camera_factory.cameras]
        if cfg.store_dataset:
            store_data_grasp(i, task_info, observations, pose, cfg.dataset_directory)
        if cfg.debug:
            image_copy = copy.deepcopy(observations[0]['rgb'])
            draw_pose(observations[0]['extrinsics'], pose, observations[0]['intrinsics'], image_copy)
            cv2.imshow('rgb', image_copy)
            depth_copy = copy.deepcopy(observations[0]['depth'])
            # rescale for visualization
            depth_copy = depth_copy / 2.0
            cv2.imshow('depth', depth_copy)
            key_pressed = cv2.waitKey(0)
            if key_pressed == ord('q'):
                break
            env.spawn_coordinate_frame(pose)
            action = Affine.from_matrix(pose)
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
