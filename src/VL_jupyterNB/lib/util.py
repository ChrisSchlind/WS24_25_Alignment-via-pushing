import numpy as np


def test_agent(agent, env, max_steps=30, render=False):
    obs = env.reset()
    path = [env.agent_current_pos]
    cumulated_reward = 0.0
    done = False
    n_steps = 0
    while not done and n_steps < max_steps:
        if render:
            env.render()
        action = agent.act(np.array([obs]), explore=False)
        obs, r, done = env.step(action)
        path.append(env.agent_current_pos)
        cumulated_reward += r
        n_steps += 1
    if render:
        env.render()
    return cumulated_reward, path


def compute_avg_return(env, agent, num_episodes=1, max_steps=200, render=False):
    total_return = 0.0
    for _ in range(num_episodes):
        episode_return, _ = test_agent(agent, env, max_steps=max_steps, render=render)
        total_return += episode_return
    return total_return / num_episodes


def train_dqn(env, agent, replay_buffer, n_epochs=100, n_steps=200, max_step_per_episode=100,
              learn_steps=128, batch_size=128, eval_after_steps=10,
              max_eval_steps=20, n_eval_episodes=10):
    for i in range(n_epochs):
        obs = env.reset()
        episode_steps = 0
        for _ in range(n_steps):
            action = agent.act(np.array([obs]))
            new_obs, r, done = env.step(action)
            replay_buffer.put(obs, action, r, new_obs, done)
            obs = new_obs
            episode_steps += 1
            if done or episode_steps >= max_step_per_episode:
                obs = env.reset()
                episode_steps = 0
        loss = 0
        for j in range(learn_steps):
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            l = agent.learn(states, actions, rewards, next_states, dones)
            loss += l
        if (i + 1) % eval_after_steps == 0:
            avg_return = compute_avg_return(env, agent, max_steps=max_eval_steps, num_episodes=n_eval_episodes)
            print(f'epoch {i + 1}, loss {loss / batch_size}, avg_return {avg_return}')
        agent.target_update()


def render_path(env, path):
    states = env.render(show_render=False)
    for position in path:
        states[tuple(position)] = 3
    print(states)
