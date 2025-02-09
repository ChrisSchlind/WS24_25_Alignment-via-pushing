import sys
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import tensorflow as tf
import cv2

from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine
from train_DQN import ConvDQN


@hydra.main(version_base=None, config_path="config", config_name="DQN")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # Setup environment
    bullet_client = setup_bullet_client(cfg.render)
    env = instantiate(cfg.env, bullet_client=bullet_client)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)

    # Load trained model
    action_dim = 2
    model = ConvDQN(action_dim)
    model.load_weights(cfg.model_path)

    # Run episodes
    for episode in range(cfg.num_test_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        for step in range(cfg.max_steps_per_episode):
            # Get action from model
            rgb_image, depth_image = state
            state_tensor = tf.convert_to_tensor([np.concatenate((rgb_image, depth_image), axis=-1)], dtype=tf.float32)
            cont_action = model(state_tensor)[0].numpy()  # shape (2,)
            # cont_action in [-1, 1], environment expects [-1,1]
            next_state, reward, done, _ = env.step(cont_action)
            episode_reward += reward

            # Update state
            state = next_state

            # Render environment
            if cfg.render:
                env.render()
                cv2.waitKey(cfg.render_delay)

            if done:
                break

            logger.info(f"Episode {episode}: Reward = {episode_reward}")

    env.close()


if __name__ == "__main__":
    main()
