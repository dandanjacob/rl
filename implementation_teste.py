'''
    Teste de implementação do ambiente Assult do Atari Learning Environment (ALE)
'''

import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
########################## ALGORITMO RL ##########################
#tendo o ambiente e a rede neural definidos, podemos implementar o algoritmo de aprendizado por reforço
#definindo os hiperparâmetros
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
n_observations = height * width * channels

#criando a rede neural e o otimizador
policy_net = DQN(n_observations, actions)
target_net = DQN(n_observations, actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=0.0001)

#criando a memória de replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
memory = deque(maxlen=10000)

#criando a função de seleção de ação
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(actions)]], dtype=torch.long)
    
#criando a função de otimização
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

########################## TREINAMENTO ##########################

#treinando a rede neural
num_episodes = 1000
for i_episode in range(num_episodes):
    # Inicializa o ambiente e o estado
    env.reset()
    last_screen = env.render(mode='rgb_array')
    current_screen = env.render(mode='rgb_array')
    state = current_screen - last_screen
    
    for t in count():
        # Seleciona e executa uma ação
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], dtype=torch.float)
        
        # Observa a nova tela
        last_screen = current_screen
        current_screen = env.render(mode='rgb_array')
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
            
        # Armazena a transição na memória
        memory.append(Transition(state, action, next_state, reward))
        
        # Move para o próximo estado
        state = next_state
        
        # Realiza uma otimização de uma amostra da memória
        optimize_model()
        if done:
            break
            
    # Atualiza a rede neural alvo, copiando os parâmetros da rede neural
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Treinamento concluído') 
