'''
    Teste de implementação do ambiente Assult do Atari Learning Environment (ALE)
'''

import gym

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
