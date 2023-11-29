#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 11: Reinforcement Learning
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from gym import envs, make, Wrapper, spaces, RewardWrapper, wrappers
from tqdm import tqdm
from collections import deque
from pyvirtualdisplay import Display

environments = envs.registry
environments_list = []
print(len(environments))
for environment in tqdm(environments):
    try:
        env = make(environment)
        environments_list.append([environment, env.observation_space, env.action_space, env.reward_range])
        env.close()
    except:
        continue

#%% MountainCar-v0
mountaincar_env = make('MountainCar-v0')
print('\u2550'*115)
print('MountainCar Environment')
print('Observation Space:', mountaincar_env.observation_space)
print('Upper Bound: {}, Lower Bound: {}'.format(mountaincar_env.observation_space.high,
                                                mountaincar_env.observation_space.low))
print('Action Space:', mountaincar_env.action_space)

initial_observation = mountaincar_env.reset(seed = 0)
print('\nInitial observation:', initial_observation)
# Get the new observation space after taking a random action
new_observation, rewards, done, info, _ = mountaincar_env.step(mountaincar_env.action_space.sample())
print('New observation:', new_observation)
mountaincar_env.close()

#%% Physics Engine (LunarLander-v2)
lunarlander_env = make('LunarLander-v2', render_mode = 'rgb_array')
observation = lunarlander_env.reset()
image = lunarlander_env.render()
plt.figure()
plt.imshow(image)
lunarlander_env.close()

#%% Classic Control (CartPole-v1)
cartpole_env = make('CartPole-v1', render_mode = 'rgb_array')
observation = cartpole_env.reset()
image = cartpole_env.render()
plt.figure()
plt.imshow(image)
cartpole_env.close()

#%% Atari (SpaceInvaders-v0)
atari_env = make('SpaceInvaders-v0', render_mode = 'rgb_array')
observation = atari_env.reset()
image = atari_env.render()
plt.figure()
plt.imshow(image)
atari_env.close()

#%% Breakout-v0
breakout_env = make('Breakout-v0', render_mode = 'rgb_array')
observation = breakout_env.reset()
# Initialize a list to store the state space at each and every step
frames = []
done = False
for _ in range(300):
    frames.append(breakout_env.render())
    obs, rewards, done, _, _ = breakout_env.step(breakout_env.action_space.sample())
    if done:
        break

def animate(i):
    patch.set_data(frames[i])

plt.figure()
patch = plt.imshow(frames[0])
plt.axis('off')
breakout_animation = animation.FuncAnimation((plt.gcf()), animate, frames = len(frames), interval = 10)
breakout_animation.save('random_breakout_agent.gif', writer = 'pillow')
breakout_env.close()

#%% BreakoutNoFramesskip-v4 Using Wrappers
class ConcatObservations(Wrapper):
    def __init__(self, environment, n):
        Wrapper.__init__(self, environment)
        self.n = n
        self.frames = deque([], maxlen = n)
        self.observation_space = spaces.Box(
            low = 0, high = 255, shape = ((n, ) + environment.observation_space.shape),
            dtype = environment.observation_space.dtype)

    def reset(self):
        observation = self.environment.reset()
        for _ in range(self.n):
            self.frames.append(observation)
        return self._get_observation()

    def step(self, action):
        observation, reward, done, info, _ = self.environment.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done ,info

    def _get_observation(self):
        return np.array(self.frames)

class ClippedRewards(RewardWrapper):
    def __init__(self, environment):
        RewardWrapper.__init__(self, environment)
        self.reward_range = (-10, 10)

    def reward(self, reward):
        '''
        Clip the reward to {+10, 0, -10} depending on its sign
        '''
        return reward if reward >= -10 and reward <= 10 else 10 * np.sign(reward)

breakoutnoframeskip_env_original = make('BreakoutNoFrameskip-v4')
breakoutnoframeskip_env_new = ConcatObservations(breakoutnoframeskip_env_original, 4)
print('\u2550'*115)
print('BreakoutNoFrameskip Environment')
print('Original observation space:', breakoutnoframeskip_env_original.observation_space)
print('New observation space:', breakoutnoframeskip_env_new.observation_space)

#%% Breakout-v4 Using Wrappers
breakout_env = make('Breakout-v4', render_mode = 'rgb_array')
breakout_recording = wrappers.monitoring.video_recorder.VideoRecorder(breakout_env, path = 'training.mp4',
                                                                      enabled = True)
for episode in range(50):
    observation = breakout_env.reset()
    for _ in range(5000):
        breakout_env.render()
        breakout_recording.capture_frame()
        action = breakout_env.action_space.sample()
        observation, reward, done, info, _ = breakout_env.step(action)
        if done:
            observation = breakout_env.reset()
breakout_recording.close()
breakout_recording.enabled = False
breakout_env.close()