#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 23:06:18 2025

@author: hamza61b
"""

from collections import deque
import os
import csv
import random
import numpy as np
import datetime
from gym.wrappers.record_video import RecordVideo
from racetrack_env import RaceTrackEnv
import gym
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten


#Define action space
action_space = {
        0: [-0.5], 1 : [0], 2 : [0.5]
    }

save_dir = "./logs/dqn2"
os.makedirs(save_dir, exist_ok=True)

#Define the DQN class
class DQNAgent:
    def __init__(self):
        self.lr = 0.0005
        self.batch_size = 256
        self.num_actions = 3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0 
        self.epsilon_decay =  0.9995 
        self.replay_size = 10000
        self.memory = deque(maxlen = self.replay_size)
        self.update_frequency = 20
        self.num_episode = 1000
        self.target_update_counter = 0
        
        self.model = self.model_definition()
        self.target_model = self.model_definition()
        self.target_model.set_weights(self.model.get_weights())
        
        
        
    def model_definition(self):
        model = Sequential()
        model.add(Flatten(input_shape=(2, 18, 18)))
        #model.add(Dense(256, input_shape=(2, 18, 18))) 
        #model.add((input_shape=(2, 18, 18)))
        model.add(Dense(256))
        model.add(Dense(256))
        model.add(Dense(256))
        model.add(Dense(self.num_actions, activation = "linear", name="output"))
        #model.build(input_shape=(None, 2, 18, 18))
        model.compile(loss = "mse", optimizer = Adam())
        model.summary()
        return model
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def q_vals(self, state):
        return self.target_model.predict_on_batch(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    def epsilon_greedy(self, state): 
        if np.random.random() <= self.epsilon:
            return np.random.randint(0,3)
        else:
            return np.argmax(self.q_vals(state))
            
        
    
    def train(self, env):
        
        rewards = []
        step_reward = []
        best_value = 0
        
        for episode in range(1, self.num_episode+1):
            
            ep_reward = 0
            state = env.reset()
            done = False 
            
            while not done:
                
                #choose an action
                action = self.epsilon_greedy(state)
                action = np.clip(action, 0, 2)
                #perform the action 
                next_state, reward, done,_ = env.step(action_space[action])
                
                ep_reward = ep_reward + reward 
                
                #Store in memory 
                self.remember(state, action, reward, next_state, done)
                
                if len(self.memory) >= 500:
                    
                    minibatch = random.sample(self.memory,self.batch_size)
                    # Prepare normalized input batches
                    batch_current_states = np.stack([transition[0] for transition in minibatch]) / 255
                    batch_next_states = np.stack([transition[3] for transition in minibatch]) / 255
            
                    # Predict Q-values for current and next states
                    current_qvalues = self.model.predict_on_batch(batch_current_states)
                    future_qvalues = self.model.predict_on_batch(batch_next_states)
                    
                    inputs = []
                    
                    targets = []
                    
                    for idx, transition in enumerate(minibatch): 
                        state, action, reward, next_state, done = transition
                        #action = np.clip(action, 0, 2)
                        #print(action)
                        if not done:
                            target_q = reward + self.gamma * np.max(future_qvalues[idx])
                        else: 
                            target_q = reward
                        
                        updated_qvalues = current_qvalues[idx]
                        updated_qvalues[action] = target_q
            
                        inputs.append(state)
                        targets.append(updated_qvalues)
            
                    # Train the model with updated targets
                    inputs = np.array(inputs) / 255.0
                    targets = np.array(targets)
            
                    self.model.train_on_batch(inputs, targets)
                    self.target_update_counter += 1
            
                    # Soft update: periodically sync target model weights
                    if self.target_update_counter >= self.update_frequency:
                        self.target_model.set_weights(self.model.get_weights())
                        self.target_update_counter = 0
                    
                    
                    
                
                state = next_state
                
            rewards.append(ep_reward)
            
            #running validation
            
            state = env.reset()
            val_reward = 0 
            done = False 
            
            while not done: 
                action = np.argmax(self.model.predict_on_batch(np.array([state]).reshape(-1, *state.shape)/255)[0])
                #print(action)
                action = np.clip(action, 0, 2)
                state, reward, done, _ = env.step(action_space[action])
                
                val_reward += reward
            
            step_reward.append(val_reward)
            
            #save best model
            if val_reward >= np.max([50,best_value]):
                best_value = val_reward
                self.model.save(f'{save_dir}/best.model')
            
            # save model every 100 episodes 
            if episode % 100 == 0:
                self.model.save(f'{save_dir}/checkpoint_{episode}.model')
                
            
            #epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
        self.model.save(f'{save_dir}/last.model')
            
                

def main():
    mode = "test"
    env = RaceTrackEnv()  


    
    env = RecordVideo(env, f'./videos/dqn/')

    if mode == "train":
        agent = DQNAgent()
        agent.train(env)

    elif mode == "test":
        save_dir = "./models"
        os.makedirs(save_dir, exist_ok=True)
        model = tf.keras.models.load_model(f"{save_dir}/DQN2.model")
        state = env.reset()
        done = False
        total_reward = 0
        trajectory = []

        while not done:
            action_idx = np.argmax(model.predict_on_batch(np.array(state).reshape(-1, *state.shape) / 255.0)[0])
            state, reward, done, _ = env.step(action_space[action_idx])
            total_reward += reward
            
            if hasattr(env, 'vehicle') and hasattr(env.vehicle, 'position'):
                trajectory.append(env.vehicle.position.copy())
                

        print("Total Test Reward:", total_reward)
        trajectory = np.array(trajectory)

        plt.figure(figsize=(6,6))
        plt.plot(trajectory[:,0], trajectory[:,1], label='Agent Trajectory')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Agent Trajectory during DQN Test')
        plt.grid(False)
        plt.axis('equal')
        plt.legend()
        plt.savefig("dqn_trajectory_.svg", format='svg', bbox_inches='tight',transparent=True)
        plt.show()

    env.close()


if __name__ == "__main__":
    main()

            
        

        
        
        
        
        
    


