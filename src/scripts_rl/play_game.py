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
from convert_util import convert_to_orthographic, display_orthographic


def draw_camera_direction(bullet_client, camera_pose, length=0.1):
    """Draws a line indicating the camera's direction."""
    start_pos = camera_pose.translation
    end_pos = start_pos + camera_pose.rotation @ np.array([0, 0, length])
    bullet_client.addUserDebugLine(start_pos, end_pos, [1, 0, 0], 2)


def update_observation_window(camera_factory, teletentric_camera, cfg, bullet_client):
    observations = [camera.get_observation() for camera in camera_factory.cameras]
    # Display
    image_copy = copy.deepcopy(observations[0]["rgb"])
    # Convert to rgb for visualization
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    cv2.imshow("rgb", image_copy)
    depth_copy = copy.deepcopy(observations[0]["depth"])
    # rescale for visualization
    depth_copy = depth_copy / 2.0
    cv2.imshow("depth", depth_copy)

    # Convert observations to orthographic view
    height_map, colormap = convert_to_orthographic(observations, cfg.workspace_bounds, cfg.projection_resolution)
    # Display orthographic view
    display_orthographic(height_map, colormap, cfg.workspace_bounds)

    # Display teletentric camera view
    teletentric_observation = teletentric_camera.get_observation()
    teletentric_image = cv2.cvtColor(teletentric_observation["rgb"], cv2.COLOR_BGR2RGB)
    cv2.imshow("teletentric_rgb", teletentric_image)


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
    teletentric_camera = instantiate(cfg.teletentric_camera, bullet_client=bullet_client, t_center=t_center)
    draw_camera_direction(bullet_client, teletentric_camera.pose)

    logger.info("Instantiation completed.")

    robot.home()
    task = task_factory.create_task()
    task.setup(env)

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
    start_pose = Affine(translation=[0.35, -0.25, cfg.fixed_z_height], rotation=[0, 0, 0, 1])
    start_pose = start_pose * gripper_offset
    robot.ptp(start_pose)

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
        # Continuously update the observation window until a key is pressed
        while True:
            update_observation_window(camera_factory, teletentric_camera, cfg, bullet_client)
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

                logger.info(f"Moving robot to {new_pose.translation}, {new_pose.rotation}")
                robot.ptp(new_pose)
                logger.info("Robot movement completed")

        if key_pressed == ord("r"):
            robot.ptp(start_pose)
            task.reset_env(env)
            logger.info("Environment reset completed")

    # Shut down
    task.clean(env)
    logger.info("Task cleanup completed")

    with stdout_redirected():
        bullet_client.disconnect()


if __name__ == "__main__":
    main()
