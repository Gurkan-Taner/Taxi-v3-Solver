import numpy as np
import gymnasium as gym
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class TaxiSolver:
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

    def run_user_mode(self):
        """Mode interactif permettant à l'utilisateur de choisir l'algorithme et les paramètres"""
        print("\n===== MODE UTILISATEUR =====")
        print("Choisissez un algorithme :")
        print("1. Force brute (baseline)")
        print("2. Q-Learning")
        print("3. Deep Q-Learning (DQN)")

        algo_choice = int(input("Votre choix (1-3) : "))
        training_episodes = int(input("Nombre d'épisodes d'entraînement : "))
        testing_episodes = int(input("Nombre d'épisodes de test : "))

        if algo_choice == 1:
            self.brute_force(testing_episodes)
        elif algo_choice == 2:
            # Paramètres pour Q-Learning
            alpha = float(input("Taux d'apprentissage (alpha) [0.1-1.0] : "))
            gamma = float(input("Facteur d'actualisation (gamma) [0.1-0.99] : "))
            epsilon = float(input("Epsilon initial [0.1-1.0] : "))
            min_epsilon = float(input("Epsilon minimal [0.01-0.1] : "))
            epsilon_decay = float(input("Taux de décroissance d'epsilon [0.01-0.1] : "))

            # Entraînement et test du Q-Learning
            q_table = self.train_q_learning(
                training_episodes, alpha, gamma, epsilon, min_epsilon, epsilon_decay
            )
            self.test_q_learning(q_table, testing_episodes)

        elif algo_choice == 3:
            # Paramètres pour DQN
            batch_size = int(input("Taille de batch [32-128] : "))
            learning_rate = float(input("Taux d'apprentissage [0.001-0.01] : "))
            gamma = float(input("Facteur d'actualisation (gamma) [0.95-0.99] : "))
            epsilon = float(input("Epsilon initial [0.1-1.0] : "))
            min_epsilon = float(input("Epsilon minimal [0.01-0.1] : "))
            epsilon_decay = float(
                input("Taux de décroissance d'epsilon [0.995-0.999] : ")
            )

            # Entraînement et test du DQN
            agent = DQNAgent(
                self.state_size,
                self.action_size,
                batch_size,
                learning_rate,
                gamma,
                epsilon,
                min_epsilon,
                epsilon_decay,
            )
            self.train_dqn(agent, training_episodes)
            self.test_dqn(agent, testing_episodes)

    def run_time_limited_mode(self, time_limit=60):
        """Mode avec paramètres optimisés pour résoudre le problème en un temps limité"""
        print(f"\n===== MODE TEMPS LIMITÉ ({time_limit}s) =====")
        training_episodes = int(input("Nombre d'épisodes d'entraînement : "))
        testing_episodes = int(input("Nombre d'épisodes de test : "))

        # Paramètres optimisés pour DQN
        batch_size = 64
        learning_rate = 0.001
        gamma = 0.99
        epsilon = 1.0
        min_epsilon = 0.01
        epsilon_decay = 0.998

        start_time = time.time()

        # Création et entraînement du DQN avec paramètres optimisés
        agent = DQNAgent(
            self.state_size,
            self.action_size,
            batch_size,
            learning_rate,
            gamma,
            epsilon,
            min_epsilon,
            epsilon_decay,
        )
        self.train_dqn(agent, training_episodes, max_time=time_limit)

        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 0:
            self.test_dqn(agent, testing_episodes)
        else:
            print("Temps écoulé ! Impossible d'effectuer les tests.")

    def brute_force(self, episodes):
        """Algorithme de force brute (baseline) pour comparaison"""
        print("\n===== ALGORITHME DE FORCE BRUTE =====")

        total_steps = 0
        total_rewards = 0

        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_steps = 0
            episode_reward = 0

            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                state = next_state
                episode_steps += 1
                episode_reward += reward

            total_steps += episode_steps
            total_rewards += episode_reward

        avg_steps = total_steps / episodes
        avg_rewards = total_rewards / episodes

        print(f"\nPerformance de la force brute sur {episodes} épisodes:")
        print(f"Nombre moyen d'étapes: {avg_steps:.2f}")
        print(f"Récompense moyenne: {avg_rewards:.2f}")

        return avg_steps, avg_rewards

    def train_q_learning(
        self,
        episodes,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=0.01,
    ):
        """Entraînement avec l'algorithme Q-Learning"""
        print("\n===== ENTRAÎNEMENT Q-LEARNING =====")

        # Initialisation de la Q-table
        q_table = np.zeros([self.state_size, self.action_size])

        # Métriques pour visualisation
        all_steps = []
        all_rewards = []

        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_steps = 0
            episode_reward = 0

            while not done:
                # Politique epsilon-greedy
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()  # Exploration
                else:
                    action = np.argmax(q_table[state])  # Exploitation

                # Exécution de l'action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Mise à jour de la Q-table (formule de Bellman)
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (
                    reward + gamma * next_max
                )
                q_table[state, action] = new_value

                state = next_state
                episode_steps += 1
                episode_reward += reward

            # Réduction d'epsilon
            epsilon = max(min_epsilon, epsilon - epsilon_decay)

            # Enregistrement des métriques
            all_steps.append(episode_steps)
            all_rewards.append(episode_reward)

            # Affichage périodique
            if episode % max(1, episodes // 10) == 0:
                print(
                    f"Épisode {episode}/{episodes}, Étapes: {episode_steps}, Récompense: {episode_reward}, Epsilon: {epsilon:.2f}"
                )

        # Affichage des métriques finales
        avg_steps = np.mean(all_steps[-100:])
        avg_rewards = np.mean(all_rewards[-100:])
        print(f"\nPerformance finale (moyenne sur 100 derniers épisodes):")
        print(f"Nombre moyen d'étapes: {avg_steps:.2f}")
        print(f"Récompense moyenne: {avg_rewards:.2f}")

        # Visualisation de l'apprentissage
        self.plot_learning_curves(all_steps, all_rewards, "Q-Learning")

        return q_table

    def test_q_learning(self, q_table, episodes):
        """Test de la Q-table apprise"""
        print("\n===== TEST Q-LEARNING =====")

        total_steps = 0
        total_rewards = 0

        total_steps_value = []
        total_rewards_value = []

        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_steps = 0
            episode_reward = 0

            while not done:
                action = np.argmax(q_table[state])  # Politique déterministe
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                state = next_state
                episode_steps += 1
                episode_reward += reward

                # Affichage pour quelques épisodes aléatoires
                if episode < 3:
                    print(
                        f"Épisode {episode+1}, Étape {episode_steps}: Action={action}, Récompense={reward}"
                    )
                    self.env.render()

            total_steps += episode_steps
            total_rewards += episode_reward

            if episode < 3 or episode == episodes - 1:
                print(
                    f"Épisode {episode+1}: {episode_steps} étapes, Récompense totale = {episode_reward}"
                )

            total_steps_value.append(episode_steps)
            total_rewards_value.append(episode_reward)

        avg_steps = total_steps / episodes
        avg_rewards = total_rewards / episodes

        print(f"\nPerformance de Q-Learning sur {episodes} épisodes:")
        print(f"Nombre moyen d'étapes: {avg_steps:.2f}")
        print(f"Récompense moyenne: {avg_rewards:.2f}")

        return avg_steps, avg_rewards

    def train_dqn(self, agent, episodes, max_time=None):
        """Entraînement avec l'algorithme Deep Q-Network"""
        print("\n===== ENTRAÎNEMENT DQN =====")

        # Métriques pour visualisation
        all_steps = []
        all_rewards = []

        start_time = time.time()

        for episode in range(episodes):
            state, _ = self.env.reset()
            state_one_hot = np.zeros(self.state_size)
            state_one_hot[state] = 1

            done = False
            episode_steps = 0
            episode_reward = 0

            while not done:
                # Vérification du temps écoulé (si mode temps limité)
                if max_time and (time.time() - start_time >= max_time):
                    print(f"Temps limite atteint après {episode} épisodes.")
                    break

                # Choix de l'action
                action = agent.act(state_one_hot)

                # Exécution de l'action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # One-hot encoding pour le prochain état
                next_state_one_hot = np.zeros(self.state_size)
                next_state_one_hot[next_state] = 1

                # Stockage de l'expérience et apprentissage
                agent.remember(state_one_hot, action, reward, next_state_one_hot, done)

                state = next_state
                state_one_hot = next_state_one_hot
                episode_steps += 1
                episode_reward += reward

                # Apprentissage par batch
                agent.replay()

            # Réduction d'epsilon
            agent.update_epsilon()

            # Enregistrement des métriques
            all_steps.append(episode_steps)
            all_rewards.append(episode_reward)

            # Arrêt si temps écoulé
            if max_time and (time.time() - start_time >= max_time):
                print(
                    f"Entraînement arrêté après {episode+1}/{episodes} épisodes (limite de temps)."
                )
                break

            # Affichage périodique
            if episode % max(1, episodes // 10) == 0:
                print(
                    f"Épisode {episode}/{episodes}, Étapes: {episode_steps}, Récompense: {episode_reward}, Epsilon: {agent.epsilon:.4f}"
                )

        # Affichage des métriques finales
        avg_steps = np.mean(all_steps[-100:] if len(all_steps) >= 100 else all_steps)
        avg_rewards = np.mean(
            all_rewards[-100:] if len(all_rewards) >= 100 else all_rewards
        )
        print(f"\nPerformance finale (moyenne sur les derniers épisodes):")
        print(f"Nombre moyen d'étapes: {avg_steps:.2f}")
        print(f"Récompense moyenne: {avg_rewards:.2f}")

        # Visualisation de l'apprentissage
        self.plot_learning_curves(all_steps, all_rewards, "DQN")

    def test_dqn(self, agent, episodes):
        """Test du modèle DQN appris"""
        print("\n===== TEST DQN =====")

        total_steps = 0
        total_rewards = 0

        for episode in range(episodes):
            state, _ = self.env.reset()
            state_one_hot = np.zeros(self.state_size)
            state_one_hot[state] = 1

            done = False
            episode_steps = 0
            episode_reward = 0

            while not done:
                action = agent.act_greedy(
                    state_one_hot
                )  # Politique déterministe (greedy)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # One-hot encoding pour le prochain état
                next_state_one_hot = np.zeros(self.state_size)
                next_state_one_hot[next_state] = 1

                state_one_hot = next_state_one_hot
                episode_steps += 1
                episode_reward += reward

                # Affichage pour quelques épisodes
                if episode < 3:
                    print(
                        f"Épisode {episode+1}, Étape {episode_steps}: Action={action}, Récompense={reward}"
                    )
                    self.env.render()

            total_steps += episode_steps
            total_rewards += episode_reward

            if episode < 3 or episode == episodes - 1:
                print(
                    f"Épisode {episode+1}: {episode_steps} étapes, Récompense totale = {episode_reward}"
                )

        avg_steps = total_steps / episodes
        avg_rewards = total_rewards / episodes

        print(f"\nPerformance de DQN sur {episodes} épisodes:")
        print(f"Nombre moyen d'étapes: {avg_steps:.2f}")
        print(f"Récompense moyenne: {avg_rewards:.2f}")

        return avg_steps, avg_rewards

    def benchmark(self, episodes_train=1000, episodes_test=100):
        """Comparaison des performances des différents algorithmes"""
        print("\n===== BENCHMARK DES ALGORITHMES =====")

        results = {}

        # 1. Force brute
        print("\nExécution de la force brute...")
        bf_steps, bf_rewards = self.brute_force(episodes_test)
        results["Force brute"] = {"steps": bf_steps, "rewards": bf_rewards}

        # 2. Q-Learning
        print("\nExécution de Q-Learning...")
        q_table = self.train_q_learning(
            episodes_train,
            alpha=0.1,
            gamma=0.95,
            epsilon=1.0,
            min_epsilon=0.01,
            epsilon_decay=0.01,
        )
        ql_steps, ql_rewards = self.test_q_learning(q_table, episodes_test)
        results["Q-Learning"] = {"steps": ql_steps, "rewards": ql_rewards}

        # 3. DQN
        print("\nExécution de DQN...")
        agent = DQNAgent(
            self.state_size,
            self.action_size,
            batch_size=64,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            min_epsilon=0.01,
            epsilon_decay=0.998,
        )
        self.train_dqn(agent, episodes_train)
        dqn_steps, dqn_rewards = self.test_dqn(agent, episodes_test)
        results["DQN"] = {"steps": dqn_steps, "rewards": dqn_rewards}

        # Visualisation comparative
        self.plot_benchmark_results(results)

        return results

    def plot_learning_curves(self, steps, rewards, algo_name):
        """Affiche les courbes d'apprentissage"""
        plt.figure(figsize=(12, 5))

        # Lissage pour mieux voir les tendances
        window = max(1, min(100, len(steps) // 10))
        steps_smoothed = pd.Series(steps).rolling(window=window, min_periods=1).mean()
        rewards_smoothed = (
            pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        )

        plt.subplot(1, 2, 1)
        plt.plot(steps, alpha=0.3, color="blue")
        plt.plot(steps_smoothed, linewidth=2, color="blue")
        plt.title(f"Nombre d'étapes par épisode - {algo_name}")
        plt.xlabel("Épisode")
        plt.ylabel("Étapes")

        plt.subplot(1, 2, 2)
        plt.plot(rewards, alpha=0.3, color="green")
        plt.plot(rewards_smoothed, linewidth=2, color="green")
        plt.title(f"Récompense par épisode - {algo_name}")
        plt.xlabel("Épisode")
        plt.ylabel("Récompense")

        plt.tight_layout()
        plt.show()

    def plot_benchmark_results(self, results):
        """Affiche les résultats comparatifs des algorithmes"""
        algos = list(results.keys())
        steps = [results[algo]["steps"] for algo in algos]
        rewards = [results[algo]["rewards"] for algo in algos]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.bar(algos, steps, color=["gray", "blue", "green"])
        plt.title("Nombre moyen d'étapes par algorithme")
        plt.ylabel("Étapes")

        for i, v in enumerate(steps):
            plt.text(i, v + 5, f"{v:.1f}", ha="center")

        plt.subplot(1, 2, 2)
        plt.bar(algos, rewards, color=["gray", "blue", "green"])
        plt.title("Récompense moyenne par algorithme")
        plt.ylabel("Récompense")

        for i, v in enumerate(rewards):
            plt.text(i, v + 0.5, f"{v:.1f}", ha="center")

        plt.tight_layout()
        plt.show()


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        batch_size=64,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=0.998,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size
        self.gamma = gamma  # Facteur d'actualisation
        self.epsilon = epsilon  # Exploration rate
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """Construction du réseau de neurones pour DQN"""
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stockage de l'expérience dans la mémoire de replay"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choix d'une action selon la politique epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def act_greedy(self, state):
        """Choix de la meilleure action (politique déterministe)"""
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        """Apprentissage à partir d'un batch d'expériences"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )

            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target

            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

    def update_epsilon(self):
        """Diminution du taux d'exploration"""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


def main():
    print("=== PROJET TAXI-V3 - APPRENTISSAGE PAR RENFORCEMENT ===")

    solver = TaxiSolver()

    print("\nChoisissez un mode :")
    print("1. Mode utilisateur (configuration manuelle)")
    print("2. Mode temps limité (paramètres optimisés)")
    print("3. Benchmark (comparaison des algorithmes)")

    mode = int(input("Votre choix (1-3) : "))

    if mode == 1:
        solver.run_user_mode()
    elif mode == 2:
        time_limit = int(input("Temps limite en secondes : "))
        solver.run_time_limited_mode(time_limit)
    elif mode == 3:
        training_episodes = int(
            input("Nombre d'épisodes d'entraînement pour le benchmark : ")
        )
        testing_episodes = int(input("Nombre d'épisodes de test pour le benchmark : "))
        solver.benchmark(training_episodes, testing_episodes)


if __name__ == "__main__":
    main()
