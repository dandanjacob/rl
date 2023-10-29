"""
    This file is used to run the assault game using the latest saved model and memory.
"""

import os
import pickle
import gym

import numpy as np
from tensorflow.keras.models import load_model # pylint: disable=import-error
from learning_implementation import DQNAgent

# Initialization
env = gym.make("ALE/Assault-v5", render_mode="rgb_array")
state_shape = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(state_shape, action_size)
BATCH_SIZE = 32
NUM_EPISODES = 1

# List to store the rewards obtained in each episode
reward_list = []

# Load the model and memory
MODEL_FILENAME = "assault_dqn_model.keras"
MEMORY_FILENAME = "memory.pkl"

for episode in range(NUM_EPISODES):
    # Loading the model
    if os.path.exists(MODEL_FILENAME):
        agent.model = load_model(MODEL_FILENAME)
        print("Model loaded from", MODEL_FILENAME)
    else:
        print("No saved model found. Training a new model for this episode.")

    # Loading the agent's memory
    if os.path.exists("MEMORY_FILENAME"):
        with open("MEMORY_FILENAME", 'rb') as f:
            agent.memory = pickle.load(f)
            print("Memory loaded successfully.")
    else:
        print("No saved memory found. Starting with an empty memory.")

    state = env.reset()[0]
    state = np.reshape(state, [1, state_shape[0], state_shape[1], state_shape[2]])
    TOTAL_REWARD = 0
    DONE = False
    while not DONE:
        # Uncomment the line below to visualize the game
        env.render()
        action = agent.act(state)
        next_state, reward, DONE, TRUNCATED, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_shape[0], state_shape[1], state_shape[2]])
        agent.remember(state, action, reward, next_state, DONE)
        state = next_state
        TOTAL_REWARD += reward
        # Stop the episode if the agent loses
        if DONE:
            reward_list.append(TOTAL_REWARD)
            break
    print(f"Episode: {episode + 1}/{NUM_EPISODES}: Reward: {TOTAL_REWARD}")
    agent.replay(BATCH_SIZE)
env.close()
print("Average reward:", np.mean(reward_list))
