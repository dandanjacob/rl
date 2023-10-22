'''
    Teste de implementação do ambiente Assult do Atari Learning Environment (ALE)
'''

import gym
import torch
import torch.nn as nn

# Verifica se o ambiente está registrado
keys = gym.envs.registry.keys()
print("ALE/Assault-v5" in keys)

# Cria o ambiente
env = gym.make("ALE/Assault-v5", render_mode="human")

# Verifica o espaço de observação e ações
height, width, channels = env.observation_space.shape
actions = env.action_space.n
print(f"Height: {height}, Width: {width}, Channels: {channels}")
print(f"Actions: {actions}")

# Rede neural para a função Q
class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(64 * 9 * 7, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Teste de execução
EPISODES = 3
for episode in range(1, EPISODES + 1):
    state = env.reset()
    DONE = False
    SCORE = 0

    # Loop de execução
    while not DONE:
        env.render()
        action = env.action_space.sample()
        n_state, reward, DONE, TRUNCATED, info = env.step(action)
        SCORE += reward

    # Exibe o resultado do episódio
    print(f"Episode: {episode}/{EPISODES}, Score: {SCORE}")
env.close()
#################################################################################
# #implementar o algoritmo DEEP Q LEARNING para o ambiente ALE/Assault-v5
# import gym
# import numpy as np
# import random
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from collections import deque
# import matplotlib.pyplot as plt
# import time

# # Cria o ambiente
# env = gym.make("ALE/Assault-v5", render_mode="human")

# # Verifica o espaço de observação e ações
# height, width, channels = env.observation_space.shape
# actions = env.action_space.n
# print(f"Height: {height}, Width: {width}, Channels: {channels}")
# print(f"Actions: {actions}")

# #cria rede neural
# def create_model():
#     model = keras.Sequential()
#     model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(height, width, channels)))
#     model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
#     model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(512, activation='relu'))
#     model.add(layers.Dense(actions, activation='linear'))
#     return model

# #cria a classe agente

