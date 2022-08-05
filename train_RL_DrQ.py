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

import pybullet as p
import numpy as np
from pathlib import Path

import utils

from DrQ import DrQAgent
from task import env
from logger import Logger
from np_replay_buffer import EfficientReplayBuffer

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
        self.agent = DrQAgent(self.cfg, self.image_shape, self.device)
        
        self._global_step = 0
        self._global_episode = 0
        
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.env = env(self.cfg, self.image_shape)
        
        self.action_shape = 1 # set by default atm, 1 action for pixel
        self.replay_buffer = EfficientReplayBuffer(self.image_shape, self.action_shape, self.cfg.replay_buffer_size, 
                                                   self.cfg.batch_size, self.cfg.nstep, self.cfg.discount, frame_stack=1)
        
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode    
        
    def evaluate(self):
        input_image = self.env.reset()
        episode_reward = 0
        start = time.time()
        
        while True:     
            with torch.no_grad():
                action, picking_pixel_y, picking_pixel_x = self.agent.act(input_image, self.global_step, eval_mode=True)
                print(f"py: {picking_pixel_y}, px: {picking_pixel_x}") 
            
            action_pixel_space = (picking_pixel_y, picking_pixel_x)
            input_image, reward, done, info = self.env.step(action_pixel_space) 
                        
            episode_reward+=reward
            
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
                
    def train(self):
        
        print("Evaluation")
        eval_reward, eval_accuracy = self.evaluate()
        if self.cfg.use_tb:
            self.log_episode(eval_reward, eval_accuracy)
        
        if self.cfg.save_snapshot:
            self.save_snapshot()
        
        train_until_step = utils.Until(self.cfg.num_train_steps)
        eval_every_episodes = utils.Every(self.cfg.eval_every_episodes)
        seed_until_step = utils.Until(self.cfg.num_seed_steps)
        
        input_image = self.env.reset()
        
        time_step = input_image.reshape((1,) + input_image.shape)
        self.replay_buffer.add(time_step, first=True)
        episode_step = 0 
        episode_reward = 0
        
        Last = False
        
        while train_until_step(self.global_step):
                        
            if Last: # reset environment
                input_image = self.env.reset()
                time_step = input_image.reshape((1,) + input_image.shape)
                self.replay_buffer.add(time_step, first=True)
                episode_step = 0 
                episode_reward = 0
                Last = False
                
            with torch.no_grad():
                action, picking_pixel_y, picking_pixel_x = self.agent.act(input_image, self.global_step, eval_mode=False)
                print(f"py: {picking_pixel_y}, px: {picking_pixel_x}") 
                
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_buffer, self.global_step)
                
                if self.cfg.use_tb:
                    self.logger.log_metrics(metrics, self.global_step, ty='train')
                
            # take env step
            action_pixel_space = (picking_pixel_y, picking_pixel_x)
            input_image, reward, done, info = self.env.step(action_pixel_space)
            
            full_action = np.array([[action.item()]])
            time_step = (input_image, full_action, reward, self.cfg.discount)
            episode_reward += reward
            self.replay_buffer.add(time_step)
            episode_step += 1
            self._global_step += 1     
            
            if done:
                Last=True
                self._global_episode += 1
    
                if eval_every_episodes(self.global_episode):
                    print("Evaluation")
                    eval_reward, eval_accuracy = self.evaluate()
                    
                    if self.cfg.use_tb:
                        self.log_episode(eval_reward, eval_accuracy)
                    
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                
    def eval_only(self):
        for _ in range(self.cfg.eval_only_iterations):
            eval_reward, accuracy = self.env.eval_episode(self.agent)
            
            if self.cfg.use_tb:
                self.log_evaluation(eval_reward, accuracy)

            print(f"Evaluation reward: {eval_reward}")
            
    def log_episode(self, eval_reward, accuracy):
        with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
            log('reward_agent', eval_reward)
            n_boxes = len(accuracy)
            for i in range(n_boxes):
                agent_accuracy = np.array(accuracy[i])
                if len(agent_accuracy)==0: agent_accuracy = np.nan
                log(f'box_{i}_accuracy_agent', agent_accuracy)
        
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
                
@hydra.main(config_path='cfgs_DrQ', config_name='config')
def main(cfg):
    from train_RL_DrQ import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    if cfg.evaluate_only:
        print("EVALUATE ONLY")
        parent_dir = root_dir.parents[2]
        snapshot = parent_dir / cfg.path2policy_eval
        assert snapshot.exists()
        workspace.load_snapshot(snapshot)
        workspace.eval_only()
    
    elif cfg.train_from_imitation:
        print("TRAIN FROM IMITATION")
        parent_dir = root_dir.parents[2]
        snapshot = parent_dir / cfg.path2policy_imit
        assert snapshot.exists()
        workspace.load_snapshot(snapshot)
        workspace.train()
        
    else:
        print("TRAIN FROM SCRATCH")
        workspace.train()

if __name__ == '__main__':
    main()


