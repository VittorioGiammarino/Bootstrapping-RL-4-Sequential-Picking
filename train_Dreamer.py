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
import functools

import pybullet as p
import numpy as np
from pathlib import Path

from utils_folder import utils_dreamer as utils

from sequential_picking_task.task import env
from logger_folder.logger import Logger

from agents.dreamerv2 import DreamerV2Agent

torch.backends.cudnn.benchmark = True

def make_dataset(episodes, config):
  generator = utils.sample_episodes(episodes, config.batch_length_training, config.oversample_ends)
  dataset = utils.from_generator(generator, config.batch_size_training)
  return dataset

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.setup()

        self.train_dataset = make_dataset(self.train_eps, self.cfg)
        self.eval_dataset = make_dataset(self.eval_eps, self.cfg)

        robot_workspace = self.env.kuka.workspace
        self.action_shape = (1,)
        self.action_pixel_shape = (self.cfg.image_width*self.cfg.image_height)

        self.agent = DreamerV2Agent(self.image_shape, self.action_shape, self.action_pixel_shape, self.cfg.device, 
                                    self.train_dataset, self.traindir, self.cfg).to(self.cfg.device)

        self.agent.requires_grad_(requires_grad=False)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):

        if self.cfg.from_segm:
            self.cfg.n_channels = 1

        self.image_shape = (self.cfg.image_width, self.cfg.image_height, self.cfg.n_channels)
        assert self.cfg.image_width == self.cfg.image_height

        print('Create env')

        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.env = env(self.cfg, self.image_shape)

        self.traindir = self.work_dir / 'train_eps'
        self.traindir.mkdir(parents=True, exist_ok=True)

        self.evaldir = self.work_dir / 'eval_eps'
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.train_eps = utils.load_episodes(self.traindir, limit=self.cfg.dataset_size)
        self.eval_eps = utils.load_episodes(self.evaldir, limit=1)

        self._episode_eval = None
        self._episode_train = None

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    def process_episode(self, mode, episode):
        directory = dict(train=self.traindir, eval=self.evaldir)[mode]
        cache = dict(train=self.train_eps, eval=self.eval_eps)[mode]
        filename = utils.save_episodes(directory, [episode])[0]
        length = len(episode['reward']) - 1
        score = float(episode['reward'].astype(np.float64).sum())
        video = episode['image']
        if mode == 'eval':
            cache.clear()
        if mode == 'train' and self.cfg.dataset_size:
            total = 0
            for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
                if total <= self.cfg.dataset_size - length:
                    total += len(ep['reward']) - 1
                else:
                    del cache[key]
        cache[str(filename)] = episode

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self.cfg.precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self.cfg.precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)

    def evaluate(self):
        step, num_episode, total_reward, number_picks, out_workspace= 0, 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        start = time.time()

        while eval_until_episode(num_episode):
            input_image, input_segm = self.env.reset()
            xyz = self.env.xyz_resized
            transition = {}

            if self.cfg.from_segm:
                obs = input_segm
            else:
                obs = input_image

            transition['image'] = obs
            transition['reward'] = 0.0
            transition['discount'] = 1.0
            self._episode_eval = [transition]

            done = False
            agent_state = None

            while not done:

                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, picking_pixel_y, picking_pixel_x, agent_state = self.agent.act(obs, agent_state, xyz, self.global_step, eval_mode=True)
                    print(f"py: {picking_pixel_y}, px: {picking_pixel_x}") 

                action_pixel_space = (picking_pixel_y, picking_pixel_x)
                input_image, input_segm, reward, done, info = self.env.step(action_pixel_space)
                xyz = self.env.xyz_resized
                transition = {}

                if self.cfg.from_segm:
                    obs = input_segm
                else:
                    obs = input_image

                transition['image'] = obs
                transition['action'] = action
                transition['action_pixel_space'] = np.array(list(action_pixel_space))
                transition['reward'] = reward
                transition['discount'] = self.cfg.discount
                self._episode_eval.append(transition)

                total_reward += reward

                if done:
                    for key,value in self._episode_eval[1].items():
                        if key not in self._episode_eval[0]:
                            self._episode_eval[0][key] = 0*value

                    episode = {k: [t[k] for t in self._episode_eval] for k in self._episode_eval[0]}
                    episode = {k: self._convert(v) for k, v in episode.items()}
                    self.process_episode("eval", episode)

            num_episode += 1
            number_picks += info["num_picked_boxes"]
            out_workspace += info["num_out_workspace"]
                
        end = time.time() - start
        print(f"Total Time: {end}, Total Reward: {total_reward / num_episode}")
        episode_reward = total_reward/num_episode
        avg_picks_per_episode = number_picks/num_episode
        avg_out_workspace_per_episode = out_workspace/num_episode
        
        return episode_reward, avg_picks_per_episode, avg_out_workspace_per_episode

    def train(self):

        print("Evaluation")
        eval_reward, avg_picks_per_episode, avg_out_workspace_per_episode = self.evaluate()
        if self.cfg.use_tb:
            self.log_episode(eval_reward, avg_picks_per_episode, avg_out_workspace_per_episode)
        
        if self.cfg.save_snapshot:
            self.save_snapshot()

        # predicates
        train_until_step = utils.Until(self.cfg.num_train_steps)
        seed_until_step = utils.Until(self.cfg.num_seed_steps)
        eval_every_episodes = utils.Every(self.cfg.eval_every_episodes)
        should_pretrain = utils.Once()

        input_image, input_segm = self.env.reset()
        xyz = self.env.xyz_resized
        transition = {}

        if self.cfg.from_segm:
            obs = input_segm
        else:
            obs = input_image

        transition['image'] = obs
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        self._episode_train = [transition]

        episode_step = 0
        episode_reward = 0

        done = False
        agent_state = None
        
        Last = False

        while train_until_step(self.global_step):

            if Last:
                # reset env
                input_image, input_segm = self.env.reset()
                xyz = self.env.xyz_resized
                transition = {}

                if self.cfg.from_segm:
                    obs = input_segm
                else:
                    obs = input_image

                transition['image'] = obs
                transition['reward'] = 0.0
                transition['discount'] = 1.0
                self._episode_train = [transition]

                agent_state = None
                episode_step = 0
                episode_reward = 0
                Last = False

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action, picking_pixel_y, picking_pixel_x, agent_state = self.agent.act(obs, agent_state, xyz, self.global_step, eval_mode=False)
                print(f"py: {picking_pixel_y}, px: {picking_pixel_x}") 

            # try to update the agent
            if not seed_until_step(self.global_step):

                if should_pretrain():
                    metrics = self.agent.pretrain(self.cfg.pretrain)

                    if self.cfg.use_tb:
                        self.logger.log_metrics(metrics, self.global_step, ty='train')

                metrics = self.agent.update(self.global_step)

                if self.cfg.use_tb:
                    self.logger.log_metrics(metrics, self.global_step, ty='train')

            # take env step
            action_pixel_space = (picking_pixel_y, picking_pixel_x)
            input_image, input_segm, reward, done, info = self.env.step(action_pixel_space)
            xyz = self.env.xyz_resized
            
            if self.cfg.from_segm:
                obs = input_segm
            else:
                obs = input_image

            transition['image'] = obs
            transition['action'] = action
            transition['action_pixel_space'] = np.array(list(action_pixel_space))
            transition['reward'] = reward
            transition['discount'] = self.cfg.discount
            self._episode_train.append(transition)

            episode_reward += reward
            episode_step += 1
            self._global_step += 1

            if done:
                Last=True
                self._global_episode += 1

                for key,value in self._episode_train[1].items():
                    if key not in self._episode_train[0]:
                        self._episode_train[0][key] = 0*value

                episode = {k: [t[k] for t in self._episode_train] for k in self._episode_train[0]}
                episode = {k: self._convert(v) for k, v in episode.items()}
                self.process_episode("train", episode)
    
                if eval_every_episodes(self.global_episode):
                    print("Evaluation")
                    eval_reward, avg_picks_per_episode, avg_out_workspace_per_episode = self.evaluate()
                    
                    if self.cfg.use_tb:
                        self.log_episode(eval_reward, avg_picks_per_episode, avg_out_workspace_per_episode)
                    
                    if self.cfg.save_snapshot:
                        self.save_snapshot()

    def log_episode(self, eval_reward, avg_picks_per_episode, avg_out_workspace_per_episode):
        with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
            log('avg_out_workspace_per_episode', avg_out_workspace_per_episode)
            log('reward_agent', eval_reward)
            log('avg_picks_per_episode', avg_picks_per_episode)

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', '_global_step']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

@hydra.main(config_path='config_folder', config_name='config_dreamer')
def main(cfg):
    from train_Dreamer import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    workspace.train()

if __name__ == '__main__':
    main()