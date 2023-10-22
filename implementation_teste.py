'''
    Teste de implementação do ambiente Assult do Atari Learning Environment (ALE)
'''

import gym
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from gym.wrappers import AtariPreprocessing

########################## AMBIENTE ##########################
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

########################## REDE NEURAL ##########################
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
    
########################## ALGORITMO RL ##########################

# Definir hiperparâmetros
learning_rate = 0.001
discount_factor = 0.99
epsilon = 0.1
EPISODES = 3

# Inicializar a rede neural Q e otimizador
q_network = QNetwork(num_actions=env.action_space.n)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

########################## TESTE DE EXECUÇÃO ##########################
# ORIGINAL DO BRENO
# for episode in range(1, EPISODES + 1):
#     state = env.reset()
#     DONE = False
#     SCORE = 0

#     # Loop de execução
#     while not DONE:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, DONE, TRUNCATED, info = env.step(action)
#         SCORE += reward

#     # Exibe o resultado do episódio
#     print(f"Episode: {episode}/{EPISODES}, Score: {SCORE}")
# env.close()
#################################################################################
#TESTE DANDAN
# Loop de treinamento
# Função para pré-processar as observações manualmente
def preprocess_observation(observation):
    # Redimensionar a observação para o tamanho desejado (neste caso, 84x84)
    observation = observation[34:194, :, :]  # Recorta a área relevante
    observation = np.mean(observation, axis=2)  # Converte para tons de cinza
    observation = observation[::2, ::2]  # Reduz a resolução pela metade
    observation = observation.astype(np.float32) / 255.0  # Normaliza os valores para o intervalo [0, 1]

    # Converter a observação para um tensor do PyTorch
    observation = torch.tensor(observation)

    # Adicionar uma dimensão de lote (batch)
    observation = observation.unsqueeze(0)

    return observation

# ...

# Loop de treinamento
for episode in range(EPISODES):
    state = env.reset()
    state = preprocess_observation(state)  # Pré-processar a observação
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = state
            q_values = q_network(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_observation(next_state)  # Pré-processar a observação


        # Calcular o alvo Q usando a equação de Bellman
        next_state = np.array(next_state)  # Converter de tuple para array NumPy
        next_state = next_state.transpose(2, 0, 1)
        next_state = next_state.astype(np.float32) / 255.0
        next_state = torch.tensor(next_state)
        next_state = next_state.unsqueeze(0)

        next_q_values = q_network(next_state)
        q_target = reward + discount_factor * next_q_values.max().item()

        # Calcular a perda de Q-learning
        state_tensor = state
        q_values = q_network(state_tensor)
        loss = nn.MSELoss()(q_values, q_target)

        # Atualizar a rede neural Q
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Avaliar o agente após o treinamento
# Implementar a avaliação do agente aqui

# Salvar o modelo treinado
torch.save(q_network.state_dict(), "q_network.pth")
