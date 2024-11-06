from os import close
import numpy as np
import gym
import random

from numpy.ma.core import argmax

env_name = "Taxi-v3"
env = gym.make(env_name)

episilon = 1.
alpha = 0.9
gamma = 0.9
episilon_decay = 0.9
min_episilon = 0.01
num_episode = 100000
max_steps = 100

q_table = np.zeros((env.observation_space.n, env.action_space.n))
#q_table = np.zeros((5,5))

def choose_action(state):
    if np.random.uniform(0, 1) < episilon:
        return env.action_space.sample()
    return np.argmax(q_table[state, :])

for episode in range(num_episode):
    state, _ = env.reset()
    done = False
    for steps in range(max_steps):
        action = choose_action(state=state)
        next_state, rewards, done, truncate, info = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = (1-alpha)*old_value + alpha*(rewards+gamma*next_max)
        state = next_state

        if done or truncate:
            break

    episilon = max(min_episilon, episilon * episilon_decay)

env = gym.make(env_name, render_mode='human')

for episode in range(10):
    state, _ = env.reset()
    done = False
    for steps in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, rewards, done, truncate, info = env.step(action)
        state = next_state
        if done or truncate:
            env.render()
            print("Done")
            break

env.close()
