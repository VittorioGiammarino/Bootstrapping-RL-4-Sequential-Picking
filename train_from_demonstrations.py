#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:06:47 2022

@author: vittoriogiammarino
"""

import random
import hydra
import pickle
import cv2
import torch
import numpy as np
from pathlib import Path

import utils_folder.utils as utils

from agents.IL_learner import Learner
from sequential_picking_task.task import env
from logger_folder.logger import Logger

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.main_dir = Path(__file__).parent
        
        print(f'workspace: {self.work_dir}')
        
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = cfg.device
        
        self.load_dataset()
        self.image_shape = self.expert_states[0].shape
        
        assert self.image_shape[0] == self.cfg.image_height and self.image_shape[1] == self.cfg.image_width
        self.agent = Learner(self.cfg, self.image_shape, self.device)
        
        self.total_steps = 0
        self.total_number_of_training_steps = self.cfg.total_number_of_training_steps
        
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.env = env(self.cfg, self.image_shape)
        
        if self.cfg.evaluate_every > self.cfg.total_number_of_training_steps:
            self.cfg.evaluate_every = self.cfg.total_number_of_training_steps
        
    def load_dataset(self):
        Path2Data = self.main_dir / self.cfg.path2data
        with open(Path2Data, 'rb') as f:
            self.dataset = pickle.load(f)
            
        self.expert_states = self.dataset["image"]
        self.expert_pixels = self.dataset["action"]
        self.avg_expert_reward = np.sum(self.dataset["reward"])/np.sum(self.dataset["terminal"])
        
        self.avg_accuracy = []
        for i in range(len(self.dataset["accuracy error"])):
            self.avg_accuracy.append(np.mean(self.dataset["accuracy error"][i]))
            
        self.avg_accuracy = np.array(self.avg_accuracy)
                    
    def get_image_transform(self, theta, trans, pivot=(0, 0)):
        """Compute composite 2D rigid transformation matrix."""
        # Get 2D rigid transformation matrix that rotates an image by theta (in radians) 
        # around pivot (in pixels) and translates by trans vector (in pixels)
        pivot_t_image = np.array([[1., 0., -pivot[0]], [0., 1., -pivot[1]], [0., 0., 1.]])
        image_t_pivot = np.array([[1., 0., pivot[0]], [0., 1., pivot[1]], [0., 0., 1.]])
        transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]], [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]])
        
        return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))

    def get_random_image_transform_params(self, image_size):
        theta_sigma = 2 * np.pi / 6
        theta = np.random.normal(0, theta_sigma)
        trans_sigma = np.min(image_size) / 6
        trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
        pivot = (image_size[1] / 2, image_size[0] / 2)
        
        return theta, trans, pivot

    def perturb(self, input_image, pixels):
        """Data augmentation on images."""
        image_size = input_image.shape[:2]
        # Compute random rigid transform.
        while True:
            theta, trans, pivot = self.get_random_image_transform_params(image_size)
            transform = self.get_image_transform(theta, trans, pivot)
            transform_params = theta, trans, pivot

            # Ensure pixels remain in the image after transform.
            is_valid = True
            new_pixels = []
            new_rounded_pixels = []
            
            for pixel in pixels:
                pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

                rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
                rounded_pixel = np.flip(rounded_pixel)

                pixel = (transform @ pixel)[:2].squeeze()
                pixel = np.flip(pixel)

                in_fov_rounded = rounded_pixel[0] < image_size[0] and rounded_pixel[1] < image_size[1]
                in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]

                is_valid = is_valid and np.all(rounded_pixel >= 0) and np.all(pixel >= 0) and in_fov_rounded and in_fov
                new_pixels.append(pixel)
                new_rounded_pixels.append(rounded_pixel)
                
            if is_valid:
                break

        # Apply rigid transform to image and pixel labels.
        input_image = cv2.warpAffine(input_image, transform[:2, :], (image_size[1], image_size[0]), flags=cv2.INTER_NEAREST)
        
        return input_image, new_rounded_pixels, transform_params
    
    def get_sample(self, augment=True):
        i = np.random.choice(range(len(self.expert_states)-1))
        state = self.expert_states[i]
        pixel = self.expert_pixels[i]
        
        if not augment:
            image = state
            label = [pixel]       
        else:
            image, label, _ = self.perturb(state, [pixel])
                
        return image, label
    
    def training_step(self):
        step = self.total_steps + 1
        self.agent.train_mode()
        
        input_image, label = self.get_sample(self.cfg.augment)
        loss = self.agent.train(input_image, label)
        self.total_steps = step
        
        if self.cfg.use_tb:
            with self.logger.log_and_dump_ctx(self.total_steps, ty='train') as log:
                log('loss', loss)
                log('step', self.total_steps)
            
    def train(self):
        for step in range(self.total_number_of_training_steps):
            self.training_step()
            
            if self.total_steps % self.cfg.evaluate_every == 0:
                for _ in range(self.cfg.eval_only_iterations):
                    eval_reward, accuracy = self.env.eval_episode(self.agent)

                    if self.cfg.use_tb:                    
                        self.log_evaluation(eval_reward, accuracy)
                
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                    
                print(f"Evaluation reward: {eval_reward}")
                    
    def eval_only(self):
        for _ in range(self.cfg.eval_only_iterations):
            eval_reward, accuracy = self.env.eval_episode(self.agent)
            
            if self.cfg.use_tb:   
                self.log_evaluation(eval_reward, accuracy)
                
            print(f"Evaluation reward: {eval_reward}")
            
    def log_evaluation(self, eval_reward, accuracy):
        with self.logger.log_and_dump_ctx(self.total_steps, ty='eval') as log:
            log('reward_expert', self.avg_expert_reward)
            log('reward_agent', eval_reward)
            n_boxes = len(self.avg_accuracy)
            for i in range(n_boxes):
                agent_accuracy = np.array(accuracy[i])
                if len(agent_accuracy)==0: agent_accuracy = np.nan
                log(f'box_{i}_accuracy_agent', agent_accuracy)
                log(f'box_{i}_accuracy_expert', self.avg_accuracy[i])
        
    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_GUI={self.cfg.GUI},image_width={self.image_shape[1]},image_height={self.image_shape[0]}.pt'
        keys_to_save = ['agent', 'total_steps']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            
    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        self.agent = payload['agent']
                
@hydra.main(config_path='config_folder', config_name='config_imitation_learning')
def main(cfg):
    from train_from_demonstrations import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    if cfg.evaluate_only:
        parent_dir = root_dir.parents[2]
        snapshot = parent_dir / cfg.path2policy
        assert snapshot.exists()
        workspace.load_snapshot(snapshot)
        workspace.eval_only()
    else:
        workspace.train()

if __name__ == '__main__':
    main()

