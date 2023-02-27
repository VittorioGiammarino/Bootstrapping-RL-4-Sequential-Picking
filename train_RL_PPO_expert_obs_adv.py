#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:03:40 2022

@author: vittoriogiammarino
"""

import hydra
import torch
import torch.optim as optim
import time
import pickle

import pybullet as p
import numpy as np
from pathlib import Path

import utils_folder.utils as utils

from agents.PPO_expert_obs_adv import PPO_adv
from sequential_picking_task.task import env
from logger_folder.logger import Logger

from buffers.np_replay_buffer_obs_only import EfficientReplayBuffer as expert_obs_buffer

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.main_dir = Path(__file__).parent
        
        print(f'workspace: {self.work_dir}')
        
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = self.cfg.device
        
        self.image_shape = (self.cfg.image_width, self.cfg.image_height, self.cfg.n_channels)
        assert self.cfg.image_width == self.cfg.image_height
        self.agent = PPO_adv(self.cfg, self.image_shape, self.device)
        
        self.total_steps = 0
        self.total_episodes = 0
        self.total_grad_steps = 0
        self.total_number_of_training_steps = self.cfg.total_number_of_training_steps
        
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.env = env(self.cfg, self.image_shape)

        self.action_shape = 1 # set by default atm, 1 action for pixel
        self.replay_buffer_expert = expert_obs_buffer(self.image_shape, self.action_shape, self.cfg.replay_buffer_size, 
                                                      self.cfg.minibatch_size, nstep=1, discount=1, frame_stack=1)
        
    def single_episode(self):
        input_image = self.env.reset()
        
        time_step = 0
        episode_reward = 0
        start = time.time()
        
        while True:      
            pick_conf, action, picking_pixel_y, picking_pixel_x = self.agent.act(input_image)
            print(f"py: {picking_pixel_y}, px: {picking_pixel_x}") 
            self.states.append(input_image)
            
            action_pixel_space = (picking_pixel_y, picking_pixel_x)
            next_input_image, reward_a, done, info = self.env.step(action_pixel_space) 

            reward = self.agent.compute_reward(input_image, next_input_image, reward_a)

            input_image = next_input_image
            
            # second version
            log_pi = pick_conf.log_prob(action)
            
            action_np = action.cpu().numpy().flatten()
            log_pi_np = log_pi.cpu().numpy().flatten()
            
            value = self.agent.forward_value_function(input_image).cpu().numpy().flatten()
            
            print(f"reward: {reward_a}, done: {done}, values: {value}, action: {action_np}")
            
            self.actions.append(action_np)
            self.rewards.append(reward)
            self.log_pis.append(log_pi_np)
            self.done.append(done)
            self.values.append(value)
            
            episode_reward+=reward_a
            time_step+=1
            self.total_steps+=1
            
            if done:
                break
            
            if self.cfg.early_stop:
                if episode_reward<0:
                    for i in range(len(self.env.list_of_boxes)):
                        p.removeBody(self.env.list_of_boxes[i])
                        p.stepSimulation()
                        
                    self.env.list_of_boxes = []
                    break
            
        end = time.time() - start
        print(f"Total Time: {end}, Total Reward: {episode_reward}")
        
        return episode_reward, self.env.accuracy
                    
    def generate_rollout(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.values = []
        self.log_pis = []
        
        samples = {}
        
        with torch.no_grad():
            for _ in range(self.cfg.num_episodes_per_rollout):
                reward, accuracy = self.single_episode()
                self.total_episodes+=1
                self.log_episode(reward, accuracy)
                
            advantages, returns = self.agent.GAE(self.done, self.rewards, self.values)
            
        samples = {'obs': np.array(self.states),
                   'actions': np.array(self.actions),
                   'values': np.array(self.values),
                   'log_pis': np.array(self.log_pis),
                   'advantages': advantages,
                   'returns': returns}
        
        return samples
    
    def minibatch_to_torch(self, minibatch):
        minibatch_torch = {}
        for k,v in minibatch.items():
            if k == 'obs':
                minibatch_torch[k] = minibatch[k]
            else:
                minibatch_torch[k] = torch.tensor(v, device=self.device)
                
        return minibatch_torch
        
    def training_step(self, step):
        metrics = dict()

        samples = self.generate_rollout()
        batch_size = len(samples['values'])

        self.agent.delta = utils.schedule(self.agent.reward_schedule, step)
        
        for _ in range(self.cfg.num_epochs):
            indexes = torch.randperm(batch_size)

            if self.agent.delta > 0:
                batch_expert = next(self.replay_buffer_expert)
                obs_e, _, next_obs_e = utils.to_torch(batch_expert, self.device)

                indexes_discr = torch.randperm(batch_size-1)

                obs_a_indexes = indexes_discr[0:self.cfg.minibatch_size].numpy()
                obs_a = samples['obs'][obs_a_indexes]
                next_obs_a = samples['obs'][obs_a_indexes+1]
                obs_a, next_obs_a = utils.to_torch((obs_a, next_obs_a), self.device)

                metrics.update(self.agent.update_discriminator(obs_a, next_obs_a, obs_e, next_obs_e))

            
            for start in range(0, batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                minibatch_indexes = indexes[start:end].numpy()
                minibatch = {}
                
                for k,v in samples.items():
                    minibatch[k] = v[minibatch_indexes]
                
                minibatch = self.minibatch_to_torch(minibatch)
                metrics.update(self.agent.train(minibatch))
                self.total_grad_steps+=1
                
                if self.cfg.use_tb:
                    self.logger.log_metrics(metrics, self.total_grad_steps, ty='train')
                
    def train(self):
        for step in range(self.total_number_of_training_steps):
            self.training_step(step)
            
            if self.cfg.use_tb:
                with self.logger.log_and_dump_ctx(step, ty='train') as log:
                    log('step', step)
                
                        
    def eval_only(self):
        for _ in range(self.cfg.eval_only_iterations):
            eval_reward, accuracy = self.env.eval_episode(self.agent)
            
            if self.cfg.use_tb:
                self.log_evaluation(eval_reward, accuracy)

            print(f"Evaluation reward: {eval_reward}")
            
    def log_episode(self, eval_reward, accuracy):
        with self.logger.log_and_dump_ctx(self.total_episodes, ty='eval') as log:
            log('reward_agent', eval_reward)
            n_boxes = len(accuracy)
            for i in range(n_boxes):
                agent_accuracy = np.array(accuracy[i])
                if len(agent_accuracy)==0: agent_accuracy = np.nan
                log(f'box_{i}_accuracy_agent', agent_accuracy)
        
    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'total_steps']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            
    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
            self.agent.load_model(payload)    

    def fill_expert_buffer(self):
        Path2Data = self.main_dir / self.cfg.path2data
        with open(Path2Data, 'rb') as f:
            self.dataset = pickle.load(f) 

        expert_states = self.dataset["state"]
        input_image = self.env.reset()
        assert input_image.shape == expert_states[0].shape
        first_obs = expert_states[0]
        
        time_step = first_obs.reshape((1,) + first_obs.shape)
        self.replay_buffer_expert.add(time_step, first=True)  
        step=1

        for i in range(len(expert_states)-1):
            obs = expert_states[1+i]
            self.replay_buffer_expert.add_from_bio(obs)
            step+=1 

        print(f'Fill expert-replay Done! Total number of samples: {step}')        

@hydra.main(config_path='config_folder', config_name='config_PPO_expert_obs_adv')
def main(cfg):
    from train_RL_PPO_expert_obs_adv import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    if cfg.evaluate_only:
        print("EVALUATE ONLY")
        parent_dir = root_dir.parents[2]
        snapshot = parent_dir / cfg.path2policy_eval
        assert snapshot.exists()
        workspace.load_snapshot(snapshot)
        workspace.eval_only()
    
    else:
        print("TRAIN LEARNING REWARD FROM EXPERT")
        print("Loading expert observations")
        workspace.fill_expert_buffer()
        workspace.train()

if __name__ == '__main__':
    main()

