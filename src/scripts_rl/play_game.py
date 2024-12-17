import sys
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy
import cv2
import random

from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine


@hydra.main(version_base=None, config_path="config", config_name="play_game")
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

    # randomly select an object and area
    id = random.randint(0, len(task.push_objects) - 1)
    obj, area = task.get_object_and_area_with_same_id(id)

    # get the object and area poses
    obj_pose = env.get_pose(obj.unique_id)
    area_pose = env.get_pose(area.unique_id)
    if cfg.auto_mode:
        print("Object pose: ", obj_pose)
        print("Area pose: ", area_pose) 

    # Move robot to start position
    gripper_offset = Affine(cfg.gripper_offset.translation, cfg.gripper_offset.rotation)
    start_pose = Affine(translation=[0.35, -0.25, 0.004], rotation=[0, 0, 0, 1])
    start_pose = start_pose * gripper_offset
    robot.ptp(start_pose)


    # Define Manual control
    switch = {
        ord("a"): Affine(translation=[-0.03, 0, 0]),
        ord("d"): Affine(translation=[0.03, 0, 0]),
        ord("w"): Affine(translation=[0, -0.03, 0]),
        ord("s"): Affine(translation=[0, 0.03, 0]),
    }

    # Control settings
    key_pressed = ord("w")
    logger.info("Control settings:")
    logger.info("w: move forward")
    logger.info("s: move backward")
    logger.info("a: move left")
    logger.info("d: move right")
    logger.info("q: quit")
    logger.info("Press any key to start")
    logger.info("Movement control is relative to the camera view displayed in the opencv window")

    while key_pressed != ord("q"):
        # Display
        image_copy = copy.deepcopy(observations[0]["rgb"])
        # Convert to rgb for visualization
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        cv2.imshow("rgb", image_copy)
        depth_copy = copy.deepcopy(observations[0]["depth"])
        # rescale for visualization
        depth_copy = depth_copy / 2.0
        cv2.imshow("depth", depth_copy)
        key_pressed = cv2.waitKey(0)

        if cfg.auto_mode:
            # Move robot
            z_offset = Affine([0, 0, 0.05])
            start_action = obj_pose * z_offset * gripper_offset
            end_action = area_pose * z_offset * gripper_offset
            robot.ptp(start_action)
            robot.lin(end_action)
        else:
            current_pose = robot.get_eef_pose()     
            action = switch.get(key_pressed, None)
            if action:
                new_pose = current_pose* action
                logger.info(f"Moving robot to {new_pose.translation}")
                robot.ptp(new_pose)
                logger.info("Robot movement completed")
                      
        
    # Shut down
    task.clean(env)

    with stdout_redirected():
        bullet_client.disconnect()


if __name__ == "__main__":
    main()