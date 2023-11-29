#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 11: Reinforcement Learning
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from collections import deque
from tensorflow.keras import models, layers, optimizers

class DQN():
    def __init__(self, env_string, batch_size = 64, IMAGE_SIZE = 84, m = 4):
        self.memory = deque(maxlen = 5000)
        self.environment = gym.make(env_string, render_mode = 'rgb_array')
        input_size = self.environment.observation_space.shape[0]
        action_size = self.environment.action_space.n
        self.batch_size = batch_size
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.IMAGE_SIZE = IMAGE_SIZE
        self.m = m

        self.recording = gym.wrappers.monitoring.video_recorder.VideoRecorder(
                self.environment, 'Recordings/{}.mp4'.format(env_string))

        # Initialize the model
        self.model = models.Sequential()
        self.model.add(layers.Convolution2D(32, 8, (4, 4), activation = 'relu', padding = 'valid',
                                            input_shape = (IMAGE_SIZE, IMAGE_SIZE, m)))
        self.model.add(layers.Convolution2D(64, 4, (2, 2), activation = 'relu', padding = 'valid'))
        self.model.add(layers.Convolution2D(64, 3, (1, 1), activation = 'relu', padding = 'valid'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation = 'elu'))
        self.model.add(layers.Dense(action_size, activation = 'linear'))
        self.model.compile(loss = 'mse', optimizer = optimizers.legacy.Adam(lr = 0.01, decay = 0.01))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        training_batch = []
        label_batch = []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target_label = self.model.predict(state)
            target_label[0][action] = reward if done else reward + (
                self.gamma * np.max(self.model.predict(next_state)[0]))
            training_batch.append(state[0])
            label_batch.append(target_label[0])
        self.model.fit(np.array(training_batch), np.array(label_batch),
                       batch_size = len(training_batch), verbose = 0)

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.environment.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def preprocess_state(self, image):
        # Extracting only the important area of the image
        image_temp = tf.image.rgb_to_grayscale(image[31:195])
        image_temp = tf.image.resize(image_temp, [self.IMAGE_SIZE, self.IMAGE_SIZE],
                                     method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image_temp = tf.cast(image_temp, tf.float32)
        return image_temp[:, :, 0]

    def combine_images(self, image1, image2):
        if len(image1.shape) == 3 and image1.shape[0] == self.m:
            combined_image = np.append(image1[1:, :, :], np.expand_dims(image2, 0), axis = 2)
        else:
            combined_image = np.stack([image1] * self.m, axis = 2)
        return tf.expand_dims(combined_image, 0)

    def train(self):
        scores = deque(maxlen = 100)
        avg_scores = []
        for epoch in range(500):
            state = self.environment.reset()
            state = self.preprocess_state(state[0])
            state = self.combine_images(state, state)
            done = False
            score = 0
            while not done:
               self.environment.render()
               self.recording.capture_frame()
               action = self.choose_action(state, self.epsilon)
               next_state, reward, done, _, _ = self.environment.step(action)
               next_state = self.preprocess_state(next_state)
               next_state = self.combine_images(next_state, state)
               self.remember(state, action, reward, next_state, done)
               state = next_state
               # Decrease the epsilon after every epoch
               self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
               score += reward
            scores.append(score)
            mean_score = np.mean(scores)
            avg_scores.append(mean_score)
            if mean_score >= 10 and epoch >= 100:
                print('Ran for {} episodes. Solved after {} trials!'.format(epoch, epoch - 100))
                self.recording.close()
                self.recording.enabled = False
                return avg_scores
            if epoch % 100 == 0:
                print('[Episode {}] - Score over the last 100 episodes was {}'.format(epoch, mean_score))
            self.replay(self.batch_size)
        self.recording.close()
        self.recording.enabled = False
        self.environment.close()
        print('Unable to solve after {} episodes :('.format(epoch))
        return avg_scores

breakout_agent = DQN('BreakoutDeterministic-v4')
breakout_agent.model.summary()
agent_scores = breakout_agent.train()

plt.figure()
plt.plot(agent_scores)
plt.xlabel('Epoch')
plt.ylabel('Score')