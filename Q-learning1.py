#Depedency
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


#Creating Environment

env_name = 'MountainCar-v0'
env = gym.make(env_name, render_mode='human')
np.random.seed(0)
env.action_space.seed(0)
env.reset(seed=0)

num_action = env.action_space.n

# Discretizing the continuous state space
# n_buckets = (18, 14)  # Number of buckets for position and velocity (discretization)
# q_table = np.zeros(n_buckets + (env.action_space.n,))

# # Hyperparameters
# episilon = 1.0
# alpha = 0.9
# gamma = 0.9
# episilon_decay = 0.995
# min_episilon = 0.01
# max_episodes = 100000
# max_steps = 10000


# def discretize_state(state):
#     position, velocity = state
#     position_buckets = np.digitize(position, np.linspace(-1.2, 0.6, n_buckets[0] - 1))
#     velocity_buckets = np.digitize(velocity, np.linspace(-0.07, 0.07, n_buckets[1] - 1))
#     return position_buckets, velocity_buckets


# def _action(state):
#     if np.random.uniform(0, 1) < episilon:
#         return env.action_space.sample()
#     return np.argmax(q_table[state, :])


# for episode in range(max_episodes):
#     state, _ = env.reset()
#     state = discretize_state(state)
#     done = False

#     for steps in range(max_steps):
#         print(f"{episode+1} : {steps+1}")
#         action = _action(state)
#         next_state, reward, done, _, info = env.step(action)
#         next_state = discretize_state(next_state)

#         old_q_val = q_table[state][action]
#         next_action = np.max(q_table[next_state])
#         new_q_val = (1 - alpha) * old_q_val + alpha * (reward + gamma * next_action)
#         q_table[state][action] = new_q_val

#         state = next_state

#         if done:
#             print(f"Episode {episode + 1}: Done after {steps + 1} steps")
#             break

#     if episilon > min_episilon:
#         episilon *= episilon_decay

# print("Training completed!")

#NN architecture
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(num_action))
model.add(Activation('linear'))

# Building Agent
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=10000, window_length=1)

agent = DQNAgent(
    policy=policy, memory=memory, model=model, nb_actions=num_action,
    nb_step_warmup=10, target_model_update=1e-2
)
agent.compile(Adam(learning_rate=1e-2), metrics=['mae'])
agent.fit(env=env, visualize=True, verbose=2, nb_steps=100)
