import gymnasium as gym
import numpy as np
import tqdm
import json


def test_matrix(env, q_matrix, hyper_params):
    episodes = 50
    total_step = 0
    total_reward = 0
    max_steps = 200

    for _ in tqdm.tqdm(range(episodes), desc="Testing"):
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
    epsilon = hyper_params["epsilon_start"]

    for _ in tqdm.tqdm(range(train_episodes), desc="Training"):
        current_state, _ = env.reset()
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_matrix[current_state])

            next_state, reward, done, _, _ = env.step(action)

            q_matrix[current_state, action] = q_matrix[current_state, action] + hyper_params["learning_rate"] * (
                reward + hyper_params["discount_factor"] * np.max(q_matrix[next_state]) - q_matrix[current_state, action]
            )
            current_state = next_state

        epsilon = max(
            hyper_params["min_epsilon"],
            epsilon * hyper_params["epsilon_decay"]
        )

    return q_matrix


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    observation, info = env.reset()

    n_observations = env.observation_space.n
    n_actions = env.action_space.n

    hyper_params_list = []
    max_iter = 100
    train_episodes = 15000

    learning_rates = np.round(np.random.uniform(0.05, 0.4, max_iter), 3)
    epsilon_starts = np.round(np.random.uniform(0.1, 1.0, max_iter), 3)
    gammas = np.round(np.random.uniform(0.9, 0.99, max_iter), 3)
    epsilon_decays = np.round(np.random.uniform(0.99, 0.999, max_iter), 4)

    print(f"Testing {max_iter} hyperparameter combinations...")
    print(f"Training episodes per combination: {train_episodes}")
    
    for i in range(max_iter):
        hyper_params = {
            "learning_rate": learning_rates[i],
            "epsilon_start": epsilon_starts[i],
            "min_epsilon": 0.01,
            "epsilon_decay": epsilon_decays[i],
            "discount_factor": gammas[i],
            "train_episodes": train_episodes,
            "avg_steps": 0,
            "avg_rewards": 0,
        }

        print(f"\nIteration {i + 1}/{max_iter}")
        print(f"Parameters: lr={hyper_params['learning_rate']}, "
              f"epsilon_start={hyper_params['epsilon_start']}, "
              f"gamma={hyper_params['discount_factor']}, "
              f"epsilon_decay={hyper_params['epsilon_decay']}")

        q_matrix = train_matrix(
            n_observations=n_observations,
            n_actions=n_actions,
            env=env,
            hyper_params=hyper_params,
            train_episodes=train_episodes,
        )

        test_matrix(q_matrix=q_matrix, env=env, hyper_params=hyper_params)

        print(f"Results: Average steps: {hyper_params['avg_steps']:.2f}, "
              f"Average rewards: {hyper_params['avg_rewards']:.2f}")
        
        hyper_params_list.append(hyper_params)


    for params in hyper_params_list:
        params["performance_score"] = params["avg_rewards"] - (params["avg_steps"] / 50)
    
    hyper_params_list.sort(key=lambda x: x["performance_score"], reverse=True)

    output_file = "taxi_optimized_hyperparams.json"
    with open(output_file, "w") as f:
        json.dump(hyper_params_list, f, indent=4)

    print("\n" + "="*80)
    print("TOP 5 BEST HYPERPARAMETER COMBINATIONS:")
    print("="*80)
    
    for i, params in enumerate(hyper_params_list[:5]):
        print(f"\nRank {i+1}:")
        print(f"  Learning Rate: {params['learning_rate']}")
        print(f"  Epsilon Start: {params['epsilon_start']}")
        print(f"  Epsilon Decay: {params['epsilon_decay']}")
        print(f"  Discount Factor (Gamma): {params['discount_factor']}")
        print(f"  Average Steps: {params['avg_steps']:.2f}")
        print(f"  Average Rewards: {params['avg_rewards']:.2f}")
        print(f"  Performance Score: {params['performance_score']:.3f}")

    print(f"\nResults saved to: {output_file}")
    print("Hyperparameters are sorted by performance score (higher is better)")