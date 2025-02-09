import sys
import copy  # Add this import
import hydra
import os
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate
from bullet_env.util import setup_bullet_client
from util.plot_util import plot_action_history, plot_rewards_epsilons, plot_losses_epsilons  # Add this import
       
@hydra.main(version_base=None, config_path="config", config_name="train_DQN")
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
    logger.info(f"Pushing Env class: {pushing_env.__class__.__name__}")
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

    # Initialize DQN supervisor
    supervisor = instantiate(cfg.supervisor, env=pushing_env, workspace_bounds=cfg.workspace_bounds)
    logger.info("DQN supervisor instantiation completed.")

    # Replay buffer
    replay_buffer = instantiate(cfg.replay_buffer)
    logger.info("Replay buffer instantiation completed.")

    logger.info("Starting training.")

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
        # set image size for the teletentric camera in the environment
        pushing_env.set_image_size(agent.input_shape)

        # Reset environment and get initial state
        state = pushing_env.reset()
        episode_reward = 0

        # Adjust max steps per episode for the first few episodes to improve learning speed
        if agent.weights_path:  # if pretrained model is loaded, use max steps from config
            max_steps = cfg.max_steps_per_episode
        else:
            max_steps = min(cfg.max_steps_per_episode, (episode + 1) * 10)
        logger.debug(f"Starting episode {episode} with max steps {max_steps}.")

        # Clear step losses for the episode        
        step_losses = []

        for step in range(max_steps):
            action, absolute_movement = agent.get_action(state, supervisor)

            # Get next state using environment's step function
            next_state, reward, done, _, failed = pushing_env.step(action, absolute_movement)

            replay_buffer.put(state, action, reward, next_state, done)

            if replay_buffer.size() >= cfg.batch_size:
                loss = agent.train(replay_buffer, cfg.batch_size, cfg.train_start_size, cfg.window_size)
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
                logger.warning(f"Episode {episode} completed at step {step}. Reward = {episode_reward}")
                break

            if failed:
                logger.debug(f"Episode {episode} failed at step {step} because all objects are outside of workspace.")
                logger.warning(f"Episode {episode} failed at step {step} because all objects are outside of workspace.")
                break

        logger.debug(f"Episode {episode}: Reward = {episode_reward}")
        logger.warning(f"Episode {episode}: Reward = {episode_reward}")

        # Save rewards and epsilon for plotting
        rewards.append(episode_reward)
        epsilons.append(agent.epsilon)
        losses.append(np.mean(step_losses))
                     
        # Plot rewards and epsilon in the same graph and save in to file periodically
        if episode % cfg.plot_freq == 0 and episode > 0:
            if cfg.activate_action_history_plot:
                plot_action_history(agent.agent_actions, episode, cfg.plot_dir, cfg.model_type)

            if cfg.activate_rewards_epsilons_plot:
                plot_rewards_epsilons(rewards, epsilons, episode, cfg.plot_dir, cfg.model_type)
            
            if cfg.activate_losses_epsilons_plot:
                plot_losses_epsilons(losses, epsilons, episode, cfg.plot_dir, cfg.model_type)

        # Save model periodically
        if episode % cfg.save_freq == 0 and episode > 0:
            agent.model.save_weights(f"{cfg.model_dir}/dqn_{cfg.model_type}_episode_{episode}", save_format="tf")

    pushing_env.close()
    logger.debug("Training completed.")
    logger.warning("Training completed.")

if __name__ == "__main__":
    main()
