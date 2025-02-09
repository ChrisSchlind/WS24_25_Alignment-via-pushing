import sys
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy
import cv2
import random
from bullet_env.util import setup_bullet_client
from transform.affine import Affine
from util.game_util import draw_camera_direction, update_observation_window

@hydra.main(version_base=None, config_path="config", config_name="play_game")
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
    
    # Display camera direction
    draw_camera_direction(bullet_client, teletentric_camera.pose)

    if cfg.debug:
        logger.info("Instantiation completed.")

    # Start the game
    robot.home()
    task = task_factory.create_task()
    task.setup(pushing_env)

    # randomly select an object and area
    id = random.randint(0, len(task.push_objects) - 1)
    obj, area = task.get_object_and_area_with_same_id(id)

    # get the object and area poses
    obj_pose = pushing_env.get_pose(obj.unique_id)
    area_pose = pushing_env.get_pose(area.unique_id)
    if cfg.auto_mode:
        print("Object pose: ", obj_pose)
        print("Area pose: ", area_pose)

    # Move robot to start position
    gripper_offset = Affine(cfg.gripper_offset.translation, cfg.gripper_offset.rotation)
    start_pose = Affine(translation=[0.35, -0.25, cfg.fixed_z_height], rotation=[0, 0, 0, 1])
    start_pose = start_pose * gripper_offset
    robot.ptp(start_pose)

    update_observation_window(camera_factory, teletentric_camera, cfg)

    # Define Manual control
    switch = {
        ord("a"): Affine(translation=[-0.03, 0, 0]),
        ord("d"): Affine(translation=[0.03, 0, 0]),
        ord("w"): Affine(translation=[0, -0.03, 0]),
        ord("s"): Affine(translation=[0, 0.03, 0]),
        ord("e"): Affine(translation=[0, 0, -0.01]),
        ord("x"): Affine(translation=[0, 0, +0.01]),
    }

    # Control settings
    key_pressed = ord("w")
    logger.info("Control settings:")
    logger.info("w: move forward")
    logger.info("s: move backward")
    logger.info("a: move left")
    logger.info("d: move right")
    logger.info("e: move up")
    logger.info("x: move down")
    logger.info("r: reset environment")
    logger.info("q: quit")
    logger.info("Press any key to start")
    logger.info("Movement control is relative to the camera view displayed in the opencv window")

    while key_pressed != ord("q"):
        # Continuously update the camera-feed. This is because robot mvt is not instantaneous
        while True:
            update_observation_window(camera_factory, teletentric_camera, cfg)
            key_pressed = cv2.waitKey(1)
            if key_pressed != -1:
                break

        if cfg.auto_mode:
            # Move robot
            z_offset = Affine([0, 0, 0.05])
            start_action = obj_pose * z_offset * gripper_offset
            end_action = area_pose * z_offset * gripper_offset

            # Define fixed z height
            start_action = Affine(translation=[start_action.translation[0], start_action.translation[1], cfg.fixed_z_height]) * gripper_offset
            end_action = Affine(translation=[end_action.translation[0], end_action.translation[1], cfg.fixed_z_height]) * gripper_offset

            robot.ptp(start_action)
            robot.lin(end_action)
        else:
            current_pose = robot.get_eef_pose()
            action = switch.get(key_pressed, None)
            if action:
                new_pose = current_pose * action

                # Drive robot to fixed height and vertical stick alignment
                new_pose = Affine(translation=[new_pose.translation[0], new_pose.translation[1], cfg.fixed_z_height]) * gripper_offset

                if cfg.debug:
                    logger.info(f"Moving robot to {new_pose.translation}, {new_pose.rotation}")
                robot.ptp(new_pose)
                if cfg.debug:
                    logger.info("Robot movement completed")

        if key_pressed == ord("r"):
            robot.ptp(start_pose)
            task.reset_env(pushing_env)
            if cfg.debug:
                logger.info("Environment reset completed")

    # Shut down
    task.clean(pushing_env)
    if cfg.debug:
        logger.info("Task cleanup completed")

    pushing_env.close()
    if cfg.debug:
        logger.info("Environment closed")

if __name__ == "__main__":
    main()
