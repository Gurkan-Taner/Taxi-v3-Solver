from time import time
import gymnasium as gym
import numpy as np
from tqdm import tqdm


class QLearning:
    def __init__(self, env: gym.Env, state_size: int, action_size: int):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = []

    def train_q_learning(
        self,
        episodes,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=0.01,
    ):
        """Train Q-Learning"""
        print("\n===== ENTRAÎNEMENT Q-LEARNING =====")
        self.q_table = np.zeros([self.state_size, self.action_size])

        all_steps = []
        all_rewards = []

        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            done = False
            episode_steps = 0
            episode_reward = 0

            while not done:
                if np.random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (
                    reward + gamma * next_max
                )
                self.q_table[state, action] = new_value

                state = next_state
                episode_steps += 1
                episode_reward += reward

            epsilon = max(min_epsilon, epsilon - epsilon_decay)

            all_steps.append(episode_steps)
            all_rewards.append(episode_reward)

            if episode % max(1, episodes // 10) == 0:
                print(
                    f"Épisode {episode}/{episodes}, Étapes: {episode_steps}, Récompense: {episode_reward}, Epsilon: {epsilon:.2f}"
                )

        return all_rewards, all_steps

    def test_q_learning(self, env, episodes, start_time=None, time_limit=None):
        """Test Q-Learning"""
        print("\n===== TEST Q-LEARNING =====")

        total_steps = []
        total_rewards = []

        for episode in tqdm(range(episodes)):
            state, _ = env.reset()
            done = False
            episode_steps = 0
            episode_reward = 0

            while not done:
                if start_time and time_limit:
                    current_time = time()
                    elapsed_time = current_time - start_time
                    if elapsed_time >= time_limit:
                        print(f"Temps écoulé après {episode} épisodes de test")
                        if episode > 0:
                            avg_steps = total_steps / episode
                            avg_rewards = total_rewards / episode

                            print(f"\nPerformance de Q-Learning sur {episode} épisode:")
                            print(f"Nombre moyen d'étapes: {avg_steps:.2f}")
                            print(f"Récompense moyenne: {avg_rewards:.2f}")
                        return

                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_steps += 1
                episode_reward += reward

            total_steps.append(episode_steps)
            total_rewards.append(episode_reward)

        return total_rewards, total_steps
