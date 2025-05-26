import gymnasium as gym
import numpy as np
import tqdm
from matplotlib import pyplot as plt
import json


def test_matrix(env, q_matrix, hyper_params):
    episodes = 50
    total_step = 0
    total_reward = 0
    max_steps = 100

    for _ in tqdm.tqdm(range(episodes)):
        done = False
        episode_reward = 0
        episode_step = 0
        current_state, _ = env.reset()

        while not done and episode_step < max_steps:
            action = np.argmax(q_matrix[current_state])
            current_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            episode_step += 1

        total_step += episode_step
        total_reward += episode_reward

    hyper_params["avg_steps"] = total_step / episodes
    hyper_params["avg_rewards"] = total_reward / episodes


def train_matrix(n_observations, n_actions, train_episodes, hyper_params, env):
    q_matrix = np.zeros((n_observations, n_actions))
    epsilon = hyper_params["epsilon"]

    for _ in tqdm.tqdm(range(train_episodes)):
        current_state, _ = env.reset()
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()

            else:
                action = np.argmax(q_matrix[current_state])

            next_state, reward, done, _, _ = env.step(action)

            q_matrix[current_state, action] = (
                1.0 - hyper_params["learning_rate"]
            ) * q_matrix[current_state, action] + hyper_params["learning_rate"] * (
                reward + hyper_params["discount_factor"] * max(q_matrix[next_state])
            )
            current_state = next_state

        epsilon = max(
            hyper_params["min_epsilon"],
            epsilon * np.exp(-hyper_params["decay_rate"]),
        )

    return q_matrix


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    observation, info = env.reset()

    n_observations = env.observation_space.n
    n_actions = env.action_space.n

    hyper_params_list = []
    i = 0
    max_iter = 100
    train_episodes = 10000

    while i < max_iter:
        learning_rate = round(np.random.uniform(0.1, 0.9), 2)
        epsilon = round(np.random.uniform(0.1, 0.9), 2)
        gamma = round(np.random.uniform(0.1, 0.9), 2)

        hyper_params = {
            "learning_rate": learning_rate,
            "epsilon": epsilon,
            "min_epsilon": 0.01,
            "decay_rate": 0.01,
            "discount_factor": gamma,
            "avg_steps": 0,
            "avg_rewards": 0,
        }

        q_matrix = train_matrix(
            n_observations=n_observations,
            n_actions=n_actions,
            env=env,
            hyper_params=hyper_params,
            train_episodes=train_episodes,
        )
        print(f"hyper_param: {hyper_params}")

        test_matrix(q_matrix=q_matrix, env=env, hyper_params=hyper_params)

        print(
            f"Iteration {i + 1} ended! Average steps: {hyper_params['avg_steps']}, average rewards: {hyper_params['avg_rewards']}"
        )
        hyper_params_list.append(hyper_params)
        i += 1

    output_file = "hyper_params_list.json"

    with open(output_file, "w") as f:
        json.dump(hyper_params_list, f, indent=4)
