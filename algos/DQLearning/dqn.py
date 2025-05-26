from time import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import tqdm
import numpy as np

GAMMA = 0.99  # Facteur de réduction pour les récompenses futures
BATCH_SIZE = 64  # Taille du batch pour l'apprentissage
BUFFER_SIZE = 10000  # Taille du buffer d'expérience replay
MIN_REPLAY_SIZE = 1000  # Taille minimale du buffer avant de commencer l'apprentissage
EPSILON_START = 1.0  # Valeur initiale d'epsilon (exploration)
EPSILON_END = 0.1  # Valeur finale d'epsilon
EPSILON_DECAY = 0.995  # Taux de décroissance d'epsilon
TARGET_UPDATE_FREQ = 100  # Fréquence de mise à jour du réseau cible
LEARNING_RATE = 1e-3  # Taux d'apprentissage


class DQNAgent:
    def __init__(self, env: gym.Env, state_size: int, action_size: int):
        self.env = env
        self.observation_space = state_size
        self.action_space = action_size

        self.online_net = DQN(input_size=state_size, output_size=action_size)
        self.target_net = DQN(input_size=state_size, output_size=action_size)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)

    def _state_to_tensor(self, state):
        # One-hot encoding pour l'état
        tensor = torch.zeros(self.observation_space)
        tensor[state] = 1.0
        return tensor

    def _select_action(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = self._state_to_tensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            return torch.argmax(q_values).item()

    def train_dqn(self, episodes):
        print("\n===== ENTRAÎNEMENT DQN =====")
        all_steps = []
        all_rewards = []
        epsilon = EPSILON_START
        step = 0

        for _ in tqdm(range(episodes)):
            state, _ = self.env.reset()
            episode_steps = 0
            episode_reward = 0
            done = False
            truncated = False

            while not (done or truncated):
                action = self._select_action(state, epsilon)

                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1

                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state

                if len(self.replay_buffer) > MIN_REPLAY_SIZE and step % 4 == 0:
                    minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

                    state_batch = torch.stack(
                        [self._state_to_tensor(s) for s, _, _, _, _ in minibatch]
                    )
                    action_batch = torch.tensor(
                        [a for _, a, _, _, _ in minibatch], dtype=torch.int64
                    )
                    reward_batch = torch.tensor(
                        [r for _, _, r, _, _ in minibatch], dtype=torch.float32
                    )
                    next_state_batch = torch.stack(
                        [
                            self._state_to_tensor(s_next)
                            for _, _, _, s_next, _ in minibatch
                        ]
                    )
                    done_batch = torch.tensor(
                        [d for _, _, _, _, d in minibatch], dtype=torch.float32
                    )

                    q_values = (
                        self.online_net(state_batch)
                        .gather(1, action_batch.unsqueeze(1))
                        .squeeze(1)
                    )

                    with torch.no_grad():
                        online_next_q_values = self.online_net(next_state_batch)
                        best_actions = torch.argmax(online_next_q_values, dim=1)
                        next_q_values = (
                            self.target_net(next_state_batch)
                            .gather(1, best_actions.unsqueeze(1))
                            .squeeze(1)
                        )
                        targets = reward_batch + GAMMA * next_q_values * (
                            1 - done_batch
                        )

                    loss = nn.functional.mse_loss(q_values, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if step % TARGET_UPDATE_FREQ == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                step += 1

            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            all_rewards.append(episode_reward)
            all_steps.append(episode_steps)

        return all_rewards, all_steps

    def test_dqn(self, env, episodes, start_time=None, time_limit=None):
        print("\n===== TEST DQN =====")

        total_steps = []
        total_rewards = []

        for episode in tqdm(range(episodes)):
            state, _ = env.reset()

            episode_steps = 0
            episode_reward = 0

            done = False

            while not done:
                if start_time and time_limit:
                    current_time = time()
                    elapsed_time = current_time - start_time
                    if elapsed_time >= time_limit:
                        print(f"Temps écoulé après {episode} épisodes de test")

                        return total_rewards, total_steps

                state_tensor = self._state_to_tensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.online_net(state_tensor)
                action = torch.argmax(q_values).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_steps += 1
                episode_reward += reward

            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)

        return total_rewards, total_steps

    def save_model(self, path):
        print(f"[INFO]: saving model to '{path}'")
        torch.save(self.online_net.state_dict(), path)

    def load_model(self, path):
        self.online_net = DQN(self.observation_space, self.action_space)
        self.online_net.load_state_dict(torch.load(path, weights_only=True))
        print(f"[INFO]: model loaded from '{path}'")


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.layers(x)
