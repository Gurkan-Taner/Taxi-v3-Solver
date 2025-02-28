from time import time
import gymnasium as gym
import numpy as np
from tqdm import tqdm


class TaxiSolver:
    def __init__(self):
        print("Initialisation de l'environnement de test Taxi-V3...")
        self.env = gym.make("Taxi-v3")
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

    def run_user_mode(self):
        """Manual mod"""
        print("\n===== MODE UTILISATEUR =====")
        print("Choisissez un algorithme :")
        print("1. Brute force")
        print("2. Q-Learning")

        algo_choice = int(input("Votre choix (1-2) : "))

        if algo_choice == 1:
            testing_episodes = int(input("Nombre d'épisodes de test : "))
            self.brute_force(episodes=testing_episodes)

        elif algo_choice == 2:
            training_episodes = int(input("Nombre d'épisodes d'entraînement : "))
            testing_episodes = int(input("Nombre d'épisodes de test : "))

            alpha = float(input("Taux d'apprentissage (alpha) [0.1-1.0] : "))
            gamma = float(input("Facteur d'actualisation (gamma) [0.1-0.99] : "))
            epsilon = float(input("Epsilon initial [0.1-1.0] : "))
            min_epsilon = float(input("Epsilon minimal [0.01-0.1] : "))
            epsilon_decay = float(input("Taux de décroissance d'epsilon [0.01-0.1] : "))

            q_table = self.train_q_learning(
                training_episodes, alpha, gamma, epsilon, min_epsilon, epsilon_decay
            )
            self.test_q_learning(q_table, testing_episodes)

    def run_time_mode(self, time_limit=None):
        """Limited time mod"""
        print(f"\n===== MODE TEMPS {f'LIMITÉ ({time_limit}s)' if time_limit else "ILLIMITÉ"} =====")
        training_episodes = int(input("Nombre d'épisodes d'entraînement : "))
        testing_episodes = int(input("Nombre d'épisodes de test : "))

        # Optimized parameters
        alpha = 0.6
        gamma = 0.7
        epsilon = 0.8
        min_epsilon = 0.1
        epsilon_decay = 1e-4

        q_table = self.train_q_learning(
            episodes=training_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            min_epsilon=min_epsilon,
            epsilon_decay=epsilon_decay,
        )

        if time_limit:
            start_time = time()

            self.test_q_learning(
                q_table=q_table,
                episodes=testing_episodes,
                start_time=start_time,
                time_limit=time_limit,
            )
        else:
            self.test_q_learning(
                q_table=q_table,
                episodes=testing_episodes,
            )

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
        q_table = np.zeros([self.state_size, self.action_size])

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
                    action = np.argmax(q_table[state])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (
                    reward + gamma * next_max
                )
                q_table[state, action] = new_value

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

        avg_steps = np.mean(all_steps[-100:])
        avg_rewards = np.mean(all_rewards[-100:])
        print(f"\nPerformance finale (moyenne sur 100 derniers épisodes):")
        print(f"Nombre moyen d'étapes: {avg_steps:.2f}")
        print(f"Récompense moyenne: {avg_rewards:.2f}")

        return q_table

    def test_q_learning(self, q_table, episodes, start_time=None, time_limit=None):
        """Test Q-Learning"""
        print("\n===== TEST Q-LEARNING =====")

        total_steps = 0
        total_rewards = 0
        env = gym.make("Taxi-v3", render_mode="human")

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
                        avg_steps = total_steps / episode
                        avg_rewards = total_rewards / episode

                        print(f"\nPerformance de Q-Learning sur {episode} épisode:")
                        print(f"Nombre moyen d'étapes: {avg_steps:.2f}")
                        print(f"Récompense moyenne: {avg_rewards:.2f}")
                        return

                action = np.argmax(q_table[state])
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_steps += 1
                episode_reward += reward

            total_steps += episode_steps
            total_rewards += episode_reward

        avg_steps = total_steps / episodes
        avg_rewards = total_rewards / episodes

        print(f"\nPerformance de Q-Learning sur {episodes} épisodes:")
        print(f"Nombre moyen d'étapes: {avg_steps:.2f}")
        print(f"Récompense moyenne: {avg_rewards:.2f}")

        return avg_steps, avg_rewards

    def brute_force(self, episodes):
        """Algorithme de force brute pour comparaison"""
        print("\n===== ALGORITHME BRUTE FORCE =====")
        print("TODO")
