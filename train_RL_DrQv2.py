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

from sequential_picking_task.task import env
from logger_folder.logger import Logger
from buffers.np_replay_buffer import EfficientReplayBuffer
from buffers.np_replay_buffer_obs_only import EfficientReplayBuffer as expert_obs_buffer

def make_agent(image_shape, cfg):
    cfg.input_shape = image_shape
    return hydra.utils.instantiate(cfg)

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.main_dir = Path(__file__).parent
        
        print(f'workspace: {self.work_dir}')
        
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = self.cfg.device
        
        if self.cfg.from_segm:
            self.cfg.n_channels = 1
        
        self.image_shape = (self.cfg.image_width, self.cfg.image_height, self.cfg.n_channels)
        assert self.cfg.image_width == self.cfg.image_height
        self.agent = make_agent(self.image_shape, self.cfg.agent)
        
        self._global_step = 0
        self._global_episode = 0
        
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.env = env(self.cfg, self.image_shape)
        
        self.action_shape = 1 # set by default atm, 1 action for pixel
        self.replay_buffer = EfficientReplayBuffer(self.image_shape, self.action_shape, self.cfg.replay_buffer_size, 
                                                   self.cfg.batch_size, self.cfg.nstep, self.cfg.discount, frame_stack=1)

        self.replay_buffer_expert = expert_obs_buffer(self.image_shape, self.action_shape, self.cfg.replay_buffer_size, 
                                                    self.cfg.batch_size, self.cfg.nstep, self.cfg.discount, frame_stack=1)
        
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode    
        
    def evaluate(self):
        step, episode, total_reward, number_picks= 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        start = time.time()

        while eval_until_episode(episode):
            input_image, input_segm = self.env.reset()

            if self.cfg.from_segm:
                state = input_segm
            else:
                state = input_image
            
            while True:     
                with torch.no_grad():
                    action, picking_pixel_y, picking_pixel_x = self.agent.act(state, eval_mode=True)
                    print(f"py: {picking_pixel_y}, px: {picking_pixel_x}") 
                
                action_pixel_space = (picking_pixel_y, picking_pixel_x)
                input_image, input_segm, reward, done, info = self.env.step(action_pixel_space)
                
                if self.cfg.from_segm:
                    state = input_segm
                else:
                    state = input_image
                            
                total_reward += reward
                number_picks += info["num_picked_boxes"]
                
                if done:
                    break
                
                if self.cfg.early_stop:
                    if total_reward<0:
                        for i in range(len(self.env.list_of_boxes)):
                            p.removeBody(self.env.list_of_boxes[i])
                            p.stepSimulation()
                            
                        self.env.list_of_boxes = []
                        break

            episode+=1
                
        end = time.time() - start
        print(f"Total Time: {end}, Total Reward: {total_reward / episode}")
        episode_reward = total_reward/episode
        avg_picks_per_episode = number_picks/episode
        
        return episode_reward, avg_picks_per_episode
                
    def train(self):
        
        print("Evaluation")
        eval_reward, avg_picks_per_episode = self.evaluate()
        if self.cfg.use_tb:
            self.log_episode(eval_reward, avg_picks_per_episode)
        
        if self.cfg.save_snapshot:
            self.save_snapshot()
        
        train_until_step = utils.Until(self.cfg.num_train_steps)
        eval_every_episodes = utils.Every(self.cfg.eval_every_episodes)
        seed_until_step = utils.Until(self.cfg.num_seed_steps)
        
        input_image, input_segm = self.env.reset()

        if self.cfg.from_segm:
            state = input_segm
        else:
            state = input_image
        
        time_step = state.reshape((1,) + state.shape)
        self.replay_buffer.add(time_step, first=True)
        episode_step = 0 
        episode_reward = 0
        
        Last = False
        
        while train_until_step(self.global_step):
                        
            if Last: # reset environment
                input_image, input_segm = self.env.reset()

                if self.cfg.from_segm:
                    state = input_segm
                else:
                    state = input_image
                
                time_step = state.reshape((1,) + state.shape)

                self.replay_buffer.add(time_step, first=True)
                episode_step = 0 
                episode_reward = 0
                Last = False
                
            with torch.no_grad():
                action, picking_pixel_y, picking_pixel_x = self.agent.act(state, eval_mode=False)
                print(f"py: {picking_pixel_y}, px: {picking_pixel_x}") 
                
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_buffer, self.replay_buffer_expert, self.global_step)
                
                if self.cfg.use_tb:
                    self.logger.log_metrics(metrics, self.global_step, ty='train')
                
            # take env step
            action_pixel_space = (picking_pixel_y, picking_pixel_x)
            input_image, input_segm, reward, done, info = self.env.step(action_pixel_space)
            
            if self.cfg.from_segm:
                state = input_segm
            else:
                state = input_image
            
            full_action = np.array([[action.item()]])
            time_step = (state, full_action, reward, self.cfg.discount)
            episode_reward += reward
            self.replay_buffer.add(time_step)
            episode_step += 1
            self._global_step += 1     
            
            if done:
                Last=True
                self._global_episode += 1
    
                if eval_every_episodes(self.global_episode):
                    print("Evaluation")
                    eval_reward = self.evaluate()
                    
                    if self.cfg.use_tb:
                        self.log_episode(eval_reward)
                    
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
            
    def log_episode(self, eval_reward, avg_picks_per_episode):
        with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
            log('reward_agent', eval_reward)
            log('avg_picks_per_episode', avg_picks_per_episode)
        
    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', '_global_step']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            
    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
            self.agent.load_critic(payload)

    def fill_expert_buffer(self):
        Path2Data = self.main_dir / self.cfg.path2data
        with open(Path2Data, 'rb') as f:
            self.dataset = pickle.load(f) 

        input_image, input_segm = self.env.reset()

        if self.cfg.from_segm:
            print("learn from instance segm")
            expert_states = self.dataset["segm"]
            assert input_segm.shape == expert_states[0].shape

        else:
            print("learn from image")
            expert_states = self.dataset["image"]
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
  
@hydra.main(config_path='config_folder', config_name='config_DrQv2')
def main(cfg):
    from train_RL_DrQv2 import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    print("Loading expert observations")
    workspace.fill_expert_buffer()
    workspace.train()
        
if __name__ == '__main__':
    main()


