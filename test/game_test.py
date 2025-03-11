import gymnasium as gym
import pygame
import sys

env = gym.make("Taxi-v3", render_mode="human").env


def debug(observation, reward, terminated, truncated, info):
    """Affiche les informations de l'environnement après chaque action."""
    print("=================================")
    print(f"observation: {observation}")
    print(f"reward: {reward}")
    print(f"terminated: {terminated}")
    print(f"truncated: {truncated}")
    print(f"info: {info}")
    print("\n")


observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_z:
                    observation, reward, terminated, truncated, info = env.step(1)
                    debug(observation, reward, terminated, truncated, info)
                    env.render()

                if event.key == pygame.K_s:
                    observation, reward, terminated, truncated, info = env.step(0)
                    debug(observation, reward, terminated, truncated, info)
                    env.render()

                if event.key == pygame.K_d:
                    observation, reward, terminated, truncated, info = env.step(2)
                    debug(observation, reward, terminated, truncated, info)
                    env.render()

                if event.key == pygame.K_q:
                    observation, reward, terminated, truncated, info = env.step(3)
                    debug(observation, reward, terminated, truncated, info)
                    env.render()

                if event.key == pygame.K_p:
                    observation, reward, terminated, truncated, info = env.step(4)
                    debug(observation, reward, terminated, truncated, info)
                    env.render()

                if event.key == pygame.K_m:
                    observation, reward, terminated, truncated, info = env.step(5)
                    debug(observation, reward, terminated, truncated, info)
                    env.render()

                if event.key == pygame.K_r:
                    print("Key R has been pressed")
                    env.reset()
                    env.render()

env.close()
