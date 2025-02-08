import sys
import copy  # Add this import
import hydra
import os
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate
from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine
from bullet_env.ur10_cell import UR10Cell  # Import UR10Cell
from bullet_env.pushing_env import PushingEnv  # Add this import

from .util.plot_util import plot_actionHistory, plot_rewards_epsilons, plot_losses_epsilons  # Add this import
       
@hydra.main(version_base=None, config_path="config", config_name="DQN")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # Setup environment - copy from play_game.py
    bullet_client = setup_bullet_client(cfg.render)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)

    # Create task factory and other components first
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    t_bounds[2, 1] = t_bounds[2, 0]
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    t_center = np.mean(t_bounds, axis=1)
    teletentric_camera = instantiate(cfg.teletentric_camera, bullet_client=bullet_client, t_center=t_center, robot=robot)

    # Create environment with all components
    env = PushingEnv(
        debug=cfg.debug,
        bullet_client=bullet_client,
        robot=robot,
        task_factory=task_factory,
        teletentric_camera=teletentric_camera,
        workspace_bounds=cfg.workspace_bounds,
        movement_bounds=cfg.movement_bounds,
        step_size=cfg.step_size,
        gripper_offset=cfg.gripper_offset,
        fixed_z_height=cfg.fixed_z_height,
        absolut_movement=cfg.absolut_movement,
        distance_TCP_obj_reward_scale=cfg.distance_TCP_obj_reward_scale,
        distance_obj_area_reward_scale=cfg.distance_obj_area_reward_scale,
        iou_reward_scale=cfg.iou_reward_scale,  # Pass the parameter
        no_movement_threshold=cfg.no_movement_threshold,
        max_moves_without_positive_reward=cfg.max_moves_without_positive_reward,
        success_threshold_trans=cfg.success_threshold_trans,
        success_threshold_rot=cfg.success_threshold_rot,
        activate_distance_obj_area_reward=cfg.activate_distance_obj_area_reward,
        activate_distance_TCP_obj_reward=cfg.activate_distance_TCP_obj_reward,
        activate_iou_reward=cfg.activate_iou_reward,
        activate_moves_without_positive_reward=cfg.activate_moves_without_positive_reward,
        activate_no_movement_punishment=cfg.activate_no_movement_punishment,
        activate_objects_outside_workspace_punishment=cfg.activate_objects_outside_workspace_punishment,
        angle_obj_area_tcp_threshold=cfg.angle_obj_area_tcp_threshold,
    )

    logger.info("Instantiation completed.")
    logger.info("Starting training with following activated rewards:")
    logger.info(f"Distance TCP-Object Reward: {cfg.activate_distance_TCP_obj_reward}")
    logger.info(f"Distance Object-Area Reward: {cfg.activate_distance_obj_area_reward}")
    logger.info(f"IoU Reward: {cfg.activate_iou_reward}")
    logger.info(f"Moves without positive reward Reward: {cfg.activate_moves_without_positive_reward}")
    logger.info(f"No movement punishment Reward: {cfg.activate_no_movement_punishment}")

    # Initialize DQN agent with 2D continuous action space
    action_dim = 4  # (left, right, up, down), discrete actions
    input_shape = (84, 84, 7)  # RGB (3) + 3 * depth (1)  + TCP one-hot-encoded (1) = 7  channels
    agent = DQNAgent(
        action_dim,
        input_shape=input_shape,
        weights_path=cfg.weights_path,
        weights_dir=cfg.weights_dir,
        learning_rate=cfg.learning_rate,  # Pass the learning_rate from the config
        use_pretrained_best_model=cfg.use_pretrained_best_model,  # Pass the use_pretrained_best_model from the config
        auto_set_epsilon=cfg.auto_set_epsilon,  # Pass the auto_set_epsilon from the config
    )
    logger.info("DQN agent initialized.")
    replay_buffer = PrioritizedReplayBuffer()
    logger.info("Replay buffer initialized.")

    # Initialize reward tracking
    rewards = []
    epsilons = []
    losses = []

    # Check if the model directory exists, if not create it
    if not os.path.exists(cfg.model_dir):
        os.makedirs(cfg.model_dir)

    # Check if the plot directory exists, if not create it
    if not os.path.exists(cfg.plot_dir):
        os.makedirs(cfg.plot_dir)

    # Training loop
    for episode in range(cfg.num_episodes):
        state = env.reset()
        episode_reward = 0

        # Adjust max steps per episode for the first few episodes to improve learning speed
        if cfg.weights_path:  # if pretrained model is loaded, use max steps from config
            max_steps = cfg.max_steps_per_episode
        else:
            max_steps = min(cfg.max_steps_per_episode, (episode + 1) * 10)
        logger.debug(f"Starting episode {episode} with max steps {max_steps}.")

        # Clear step losses for the episode        
        step_losses = []

        for step in range(max_steps):
            action = agent.get_action(state)

            # Get next state using environment's step function
            next_state, reward, done, _, failed = env.step(action)

            replay_buffer.put(state, action, reward, next_state, done)

            if replay_buffer.size() >= cfg.batch_size:
                loss = agent.train(replay_buffer, cfg.batch_size)
                if loss is None:
                    loss = 0.0
            else:
                loss = 0.0

            step_losses.append(np.mean(loss))

            if step % cfg.target_update_freq == 0:
                agent.update_target()

            state = next_state
            episode_reward += reward

            if done:
                logger.debug(f"Episode {episode} completed at step {step}. Reward = {episode_reward}")
                logger.error(f"Episode {episode} completed at step {step}. Reward = {episode_reward}")
                break

            if failed:
                logger.debug(f"Episode {episode} failed at step {step} because all objects are outside of workspace.")
                logger.error(f"Episode {episode} failed at step {step} because all objects are outside of workspace.")
                break

        logger.debug(f"Episode {episode}: Reward = {episode_reward}")
        logger.error(f"Episode {episode}: Reward = {episode_reward}")

        # Save rewards and epsilon for plotting
        rewards.append(episode_reward)
        epsilons.append(agent.epsilon)
        losses.append(np.mean(step_losses))
                     
        # Plot rewards and epsilon in the same graph and save in to file periodically
        if episode % cfg.plot_freq == 0 and episode > 0:
            plot_rewards_epsilons(rewards, epsilons, episode, cfg.plot_dir)
            plot_actionHistory(agent.agent_actions, cfg.plot_dir, episode)  # Plot agent actions
            plot_losses_epsilons(losses, epsilons, episode, cfg.plot_dir)

        # Save model periodically
        if episode % cfg.save_freq == 0 and episode > 0:
            agent.model.save_weights(f"{cfg.model_dir}/dqn_episode_{episode}", save_format="tf")

    env.close()
    logger.debug("Training completed.")
    logger.error("Training completed.")


if __name__ == "__main__":
    main()
