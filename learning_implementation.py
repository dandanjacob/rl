'''
    Training a DQN agent to play the Atari game "Assault-v5"
'''

# Import the required libraries
import os
import random
from collections import deque
import pickle
import gym

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model # pylint: disable=import-error
from tensorflow.keras.layers import Dense, Flatten, Convolution2D # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam # pylint: disable=import-error

# Check if the Atari environment is installed
keys = gym.envs.registry.keys()
if "ALE/Assault-v5" not in keys:
    assert False, "Please install the atari environment by running: pip3 install gym[atari]"
else:
    print("ALE/Assault-v5" in keys)

# Make the environment
env = gym.make("ALE/Assault-v5", render_mode="rgb_array", obs_type="rgb", full_action_space=False)

# View of the environment
height, width, channels = env.observation_space.shape
actions = env.action_space.n
print(f"Height: {height}, Width: {width}, Channels: {channels}")
print(f"Actions: {actions}")

# Define the deep Q-learning agent class
class DQNAgent:
    '''
        Deep Q-learning agent
    '''
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Create the Q-network
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, DONE):
        self.memory.append((state, action, reward, next_state, DONE))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, BATCH_SIZE):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, DONE in minibatch:
            target = reward
            if not DONE:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialization
env = gym.make("ALE/Assault-v5", render_mode="rgb_array")
state_shape = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(state_shape, action_size)
BATCH_SIZE = 32
NUM_EPISODES = 5

# List to store the rewards obtained in each episode
reward_list = []

# Training the agent
MODEL_FILENAME = 'assault_dqn_model.keras'
for episode in range(NUM_EPISODES):
    # Loading the model
    if os.path.exists(MODEL_FILENAME):
        agent.model = load_model(MODEL_FILENAME)
        print("Model loaded from", MODEL_FILENAME)
    else:
        print("No saved model found. Training a new model for this episode.")

    # Loading the agent's memory
    if os.path.exists('memory.pkl'):
        with open('memory.pkl', 'rb') as f:
            agent.memory = pickle.load(f)
            print("Memory loaded successfully.")
    else:
        print("No saved memory found. Starting with an empty memory.")

    state = env.reset()[0]
    state = np.reshape(state, [1, state_shape[0], state_shape[1], state_shape[2]])
    TOTAL_REWARD = 0
    for time in range(5000):
        # Uncomment the line below to visualize the game
        env.render()
        action = agent.act(state)
        next_state, reward, DONE, TRUNCATED, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_shape[0], state_shape[1], state_shape[2]])
        agent.remember(state, action, reward, next_state, DONE)
        state = next_state
        TOTAL_REWARD += reward

        # Stop the episode when the maximum time step is reached
        if time == 499:
            DONE = True

        # Stop the episode if the agent loses
        if DONE:
            reward_list.append(TOTAL_REWARD)
            break
    print(f"Episode: {episode + 1}/{NUM_EPISODES}: Reward: {TOTAL_REWARD}")
    # Save the model after training
    agent.model.save(MODEL_FILENAME)

    with open('memory.pkl', 'wb') as f:
        pickle.dump(agent.memory, f)
    agent.replay(BATCH_SIZE)
env.close()

#print reward_list and some statistics
print(reward_list)
print(f"Average reward: {np.mean(reward_list)}")
print(f"Max reward: {np.max(reward_list)}")
print(f"Min reward: {np.min(reward_list)}")

# Save the plot of rewards over episodes
if not os.path.exists("plots"):
    os.makedirs("plots")


plt.plot(reward_list)
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards over episodes")
plt.savefig("plots/assault.png")
