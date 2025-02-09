import sys
import copy  # Add this import
import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate
from bullet_env.util import setup_bullet_client# Add this import
       
@hydra.main(version_base=None, config_path="config", config_name="play_DQN")
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

    # Create environment with all components
    pushing_env = instantiate(cfg.pushing_env, bullet_client=bullet_client, robot=robot, task_factory=task_factory, teletentric_camera=teletentric_camera)
    logger.info("Environment instantiation completed.")

    # Initialize DQN agent
    if cfg.model_type == "ResNet":
        agent = instantiate(cfg.agent_resnet)        
    elif cfg.model_type == "FCN":
        agent = instantiate(cfg.agent_fcn)
    elif cfg.model_type == "CNN":
        agent = instantiate(cfg.agent_cnn)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")
    
    logger.info(f"Agent class: {agent.__class__.__name__}")
    logger.info("DQN agent instantiation completed.")

    # Load weights for model
    agent.load_weights(cfg.weights_dir, cfg.weights_path, training=False)
    logger.info("Weights loaded.")

    #  Number of runs
    for run in range(cfg.num_runs):
        logger.info(f"Start with Run {run}.")

        # set image size for the teletentric camera in the environment
        pushing_env.set_image_size(agent.input_shape)

        # Reset environment
        state = pushing_env.reset()
        
        for step in range(cfg.max_steps):
            logger.info(f"Run {run}, Step {step}.")

            # Get action from agent
            action, absolute_movement = agent.get_action(state, supervisor=None, training=False)

            # Get next state using environment's step function
            next_state, _, done, _, failed = pushing_env.step(action, absolute_movement)
            state = next_state

            if done:
                logger.info(f"Run {run} completed at step {step}.")
                break

            if failed:
                logger.info(f"Run {run} failed at step {step} because all objects are outside of workspace.")
                break

    pushing_env.close()
    logger.debug("All Runs completed.")

if __name__ == "__main__":
    main()