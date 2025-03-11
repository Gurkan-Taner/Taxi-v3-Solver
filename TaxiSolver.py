from time import time
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from algos.QLearning.q_learning import QLearning


class TaxiSolver:
    def __init__(self):
        print("Initialisation de l'environnement de test Taxi-V3...")
        self.env = gym.make("Taxi-v3")
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        self.q_learning = QLearning(self.env, self.state_size, self.action_size)

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

            q_table = self.q_learning.train_q_learning(
                training_episodes, alpha, gamma, epsilon, min_epsilon, epsilon_decay
            )

            self.q_learning.test_q_learning(q_table, testing_episodes)

    def run_time_mode(self, time_limit=None):
        """Time mod"""
        print(
            f"\n===== MODE TEMPS {f'LIMITÉ ({time_limit}s)' if time_limit else "ILLIMITÉ"} ====="
        )
        training_episodes = int(input("Nombre d'épisodes d'entraînement : "))
        testing_episodes = int(input("Nombre d'épisodes de test : "))

        # Optimized parameters
        alpha = 0.6
        gamma = 0.7
        epsilon = 0.8
        min_epsilon = 0.1
        epsilon_decay = 1e-4

        q_table = self.q_learning.train_q_learning(
            training_episodes, alpha, gamma, epsilon, min_epsilon, epsilon_decay
        )

        if time_limit:
            start_time = time()

            self.q_learning.test_q_learning(
                q_table=q_table,
                episodes=testing_episodes,
                start_time=start_time,
                time_limit=time_limit,
            )
        else:
            self.q_learning.test_q_learning(
                q_table=q_table,
                episodes=testing_episodes,
            )

    def brute_force(self, episodes):
        """Algorithme de force brute pour comparaison"""
        print("\n===== ALGORITHME BRUTE FORCE =====")
        print("TODO")
