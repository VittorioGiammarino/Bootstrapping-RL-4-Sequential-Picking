#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:17:52 2022

@author: vittoriogiammarino
"""

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch import autograd

import utils

from models.resnet import ResNet43_8s, MLP, Discriminator
from einops.layers.torch import Rearrange
from torch.distributions.categorical import Categorical


class RandomRotateShiftsAug:
    def __init__(self, device):
        self.device = device
        
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

    def perturb(self, input_image, pixels, n_input_image):
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
        n_input_image = cv2.warpAffine(n_input_image, transform[:2, :], (image_size[1], image_size[0]), flags=cv2.INTER_NEAREST)
        
        return input_image, new_rounded_pixels, n_input_image, transform_params
    
    def get_action_from_pixels(self, pixel_label, input_img):
        label_size = input_img.shape[:2] + (1,)
        label = np.zeros(label_size)
        label[pixel_label[0][0], pixel_label[0][1], 0] = 1
        label = torch.tensor(label, dtype=torch.float32)
        label_flatten = Rearrange('h w c -> 1 (h w c)')(label)
        action = torch.argmax(label_flatten, dim=1).numpy()
        
        return action
    
    def get_pixels_from_action(self, action, input_image):
        pixels = np.unravel_index(int(action), shape=input_image.shape[:2])
        return pixels[:2]
    
    def get_data_aug(self, obs, action, nobs):       
        action_pick = action[0]   
        
        pixels = self.get_pixels_from_action(action_pick, obs)
        image, new_pixel_label, n_image, _ = self.perturb(obs, [pixels], nobs)
        picking_action = self.get_action_from_pixels(new_pixel_label, image)
        
        new_action = np.array([picking_action.item()])

        return image, new_action, n_image
    
class Critic(nn.Module):
    def __init__(self, input_shape, output_channels, device):
        super().__init__()
        
        self.device = device
        
        self.in_shape = input_shape
        max_dim = np.max(self.in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(self.in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
        
        self.critic_1 = ResNet43_8s(self.in_shape[2], output_channels)
        self.critic_2 = ResNet43_8s(self.in_shape[2], output_channels)
        
    def forward(self, input_img, softmax_bool):
        in_tens = self.resize_input_img(input_img)
        in_tens = torch.split(in_tens, 1, dim=0)
        
        logits_critic_1 = ()
        logits_critic_2 = ()
        
        for x in in_tens:
            logits_critic_1 += (self.critic_1(x),)
            logits_critic_2 += (self.critic_2(x),)
        
        logits_critic_1 = torch.cat(logits_critic_1, dim=0)
        logits_critic_2 = torch.cat(logits_critic_2, dim=0)
        
        output_critic_1 = self.pad_rearrange(logits_critic_1)
        output_critic_2 = self.pad_rearrange(logits_critic_2)
        
        if softmax_bool:
            output_critic_1 = nn.Softmax(dim=1)(output_critic_1)
            output_critic_1 = output_critic_1.detach().cpu().numpy()
            output_critic_1 = np.float32(output_critic_1).reshape(logits_critic_1.shape[1:])
            
            output_critic_2 = nn.Softmax(dim=1)(output_critic_2)
            output_critic_2 = output_critic_2.detach().cpu().numpy()
            output_critic_2 = np.float32(output_critic_2).reshape(logits_critic_2.shape[1:])
        
        return output_critic_1, output_critic_2
        
    def process_img(self, input_img):
        img = self.normalize(input_img)
        return img
        
    def normalize(self, input_img):
        img = (input_img / 255)
        return img
    
    def resize_input_img(self, input_img):
        
        if len(input_img.shape)==3:
            in_data = np.pad(input_img, self.padding, mode='constant')
            in_data_processed = self.process_img(in_data)
            in_shape = (1,) + in_data_processed.shape
            in_data_processed = in_data_processed.reshape(in_shape)
            in_tens = torch.tensor(in_data_processed, dtype=torch.float32).to(self.device)
            
        elif len(input_img.shape)==4:
            batch_size = input_img.shape[0]
            
            in_data_processed_array = np.zeros(input_img.shape, dtype=np.float32)
            
            for i in range(batch_size):
                img = input_img[i,:,:,:]
                in_data = np.pad(img, self.padding, mode='constant')
                in_data_processed = self.process_img(in_data)
                in_shape = (1,) + in_data_processed.shape
                in_data_processed = in_data_processed.reshape(in_shape)
                
                in_data_processed_array[i] = in_data_processed
                
            in_tens = torch.tensor(in_data_processed_array, dtype=torch.float32).to(self.device)
            
        return in_tens  
    
    def pad_rearrange(self, logits):
        c0 = self.padding[:2, 0]
        c1 = c0 + self.in_shape[:2]
        logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

        output = Rearrange('b h w c -> b (h w c)')(logits)
        
        return output
        

class DrQAgent_adv:
    def __init__(self, config, input_shape, device):
        
        self.config = config
        self.device = device
        self.use_tb = self.config.use_tb
        self.critic_target_tau = self.config.critic_target_tau
        self.update_every_steps = self.config.update_every_steps
        self.stddev_schedule = self.config.stddev_schedule
        
        output_channels = self.config.decoder_nc

        self.discriminator = Discriminator(input_shape, self.config.feature_dim, self.config.hidden_dim).to(self.device)
        self.reward_d_coef = self.config.reward_d_coef
        self.reward_schedule = self.config.reward_schedule
        self.delta = 1.0
        
        self.critic = Critic(input_shape, output_channels, self.device).to(self.device)
        self.critic_target = Critic(input_shape, output_channels, self.device).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        #optimizers
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        self.discriminator_opt = optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate)
        
        # data augmentation
        self.aug = RandomRotateShiftsAug(self.device)
        
        self.train()
        self.critic_target.train()
        
    def load_critic(self, payload):
        self.critic.critic_1.load_state_dict(payload['agent'].encoder.state_dict())
        self.critic.critic_2.load_state_dict(payload['agent'].encoder.state_dict())
            
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        #optimizers
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        
    def train(self, training=True):
        self.training = training 
        self.critic.train(training)
        self.discriminator.train(training)
        
    def add_noise(self, distr, stddev):
        return torch.normal(0, stddev, size=distr.shape).to(self.device)
    
    def gaussian_1d_filter(self, x=0, mx=0, sx=1):
        return (1 / (2*np.pi*sx) * torch.exp(-((x - mx)**2 / (2*sx**2)))).reshape(1,-1)
        
    def act(self, input_image, step, eval_mode=True):
                    
        if eval_mode:
            
            target_Q1, target_Q2 = self.critic_target(input_image, softmax_bool=False)         
            target_V = torch.min(target_Q1, target_Q2)
            
            pick_conf = nn.Softmax(dim=1)(target_V)
            pi = Categorical(probs = pick_conf)
            
            pick_conf = pick_conf.detach().cpu().numpy()
            pick_conf = np.float32(pick_conf).reshape(input_image.shape[:2])

            #this one works cause we are processing a single image at the time during eval mode        
            action = np.argmax(pick_conf) 
            action = np.unravel_index(action, shape=pick_conf.shape)
            
            picking_pixel = action[:2]
            picking_y = picking_pixel[0]
            picking_x = picking_pixel[1]               
            
        else:
            
            target_Q1, target_Q2 = self.critic_target(input_image, softmax_bool=False)
            target_V = torch.min(target_Q1, target_Q2) 
            pick_conf = target_V

            if self.config.gaussian_filter:

                pick_conf = pick_conf.detach().cpu().numpy()
                pick_conf = np.float32(pick_conf).reshape(input_image.shape[:2])

                action = np.argmax(pick_conf) 
                action = np.unravel_index(action, shape=pick_conf.shape)
                
                picking_pixel = action[:2]
                picking_y = picking_pixel[0]
                picking_x = picking_pixel[1]    

                stddev = utils.schedule(self.stddev_schedule, step)

                picking_y = int(np.clip(np.random.normal(picking_y, stddev), 0, input_image.shape[0]-1))
                picking_x = int(np.clip(np.random.normal(picking_y, stddev), 0, input_image.shape[1]-1))
                action = self.aug.get_action_from_pixels([[picking_y, picking_x]], input_image)

            else:

                pi = Categorical(logits = pick_conf)
                
                action = pi.sample()
                action_numpy = action.detach().cpu().numpy()
                pick_conf = pick_conf.reshape(input_image.shape[:2])
                action_reshaped = np.unravel_index(action_numpy, shape=pick_conf.shape)
                
                picking_pixel = action_reshaped[:2]
                picking_y = int(picking_pixel[0])
                picking_x = int(picking_pixel[1])

                action = action_numpy
            
        return action, picking_y, picking_x
    
    def act_batch(self, input_image):        
        target_Q1, target_Q2 = self.critic_target(input_image, softmax_bool=False)
        target_V = torch.min(target_Q1, target_Q2) 
        pick_conf = target_V
        argmax = torch.argmax(pick_conf, dim=-1)
        pi = Categorical(logits = pick_conf)
            
        return target_V, argmax, pi
        
    
    def update_critic(self, obs, action, reward, discount, next_obs):
        metrics = dict()
        
        with torch.no_grad():
            target_V, next_pick, pi = self.act_batch(next_obs)
            target_V_pick = target_V.gather(1, next_pick.reshape(-1,1))
            target_Q = reward + (discount*target_V_pick)
            
        Q1, Q2 = self.critic.forward(obs, softmax_bool=False)
        
        pick = action[:,0]
        Q1_picked = Q1.gather(1, pick.reshape(-1,1).long())
        Q2_picked = Q2.gather(1, pick.reshape(-1,1).long())
        
        critic_loss = F.mse_loss(Q1_picked, target_Q) + F.mse_loss(Q2_picked, target_Q) 
                        
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1_picked.mean().item()
            metrics['critic_q2'] = Q2_picked.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['policy_entropy'] = pi.entropy().mean().item()
                        
        self.optimizer_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.optimizer_critic.step()
        
        return metrics

    def compute_reward(self, obs_a, next_obs_a, reward_a):
        
        # encode
        with torch.no_grad():
            self.discriminator.eval()
            d = self.discriminator(obs_a, next_obs_a)
            reward_d = self.reward_d_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            
            reward = self.delta*reward_d + (1.0 - self.delta)*reward_a
    
            self.discriminator.train()
            
        return reward
    
    def compute_discriminator_grad_penalty(self, obs_e, next_obs_e, lambda_=10):
        metrics = dict()

        obs = obs_e / 255.0 - 0.5
        nobs = next_obs_e / 255.0 - 0.5
        h = self.discriminator.convnet(obs)
        nh = self.discriminator.convnet(nobs)
        h = h.reshape(h.shape[0], -1)
        nh = h.reshape(nh.shape[0], -1)
        
        h = h.detach()
        nh = nh.detach()

        expert_data = torch.cat([h, nh], dim=-1)
        expert_data.requires_grad = True
        
        d = self.discriminator.net(self.discriminator.trunk(expert_data))
        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=expert_data, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        
        if self.use_tb:
            metrics['discriminator_grad_pen'] = grad_pen.item()
        
        return grad_pen
        
    def update_discriminator(self, obs_a, next_obs_a, obs_e, next_obs_e):
        metrics = dict()
        
        agent_d = self.discriminator(obs_a, next_obs_a)
        expert_d = self.discriminator(obs_e, next_obs_e)
        
        expert_loss = F.mse_loss(expert_d, torch.ones(expert_d.size(), device=self.device))
        agent_loss = F.mse_loss(agent_d, -1*torch.ones(agent_d.size(), device=self.device))
        
        grad_pen_loss = self.compute_discriminator_grad_penalty(obs_e, next_obs_e)
        
        loss = 0.5*(expert_loss + agent_loss) + grad_pen_loss
        
        # optimize inverse models
        self.discriminator_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_opt.step()
        
        if self.use_tb:
            metrics['discriminator_loss'] = loss.item()
        
        return metrics   
    
    def update(self, replay_iter, replay_buffer_expert, step):
        metrics = dict()
        
        if step % self.update_every_steps != 0:
            return metrics

        self.delta = utils.schedule(self.reward_schedule, step)

        if self.delta > 0:
            batch_expert = next(replay_buffer_expert)
            obs_e, _, next_obs_e = utils.to_torch(batch_expert, self.device)

            batch = next(replay_iter)
            obs_a, action_a, reward_a, discount_a, next_obs_a = utils.to_torch(batch, self.device)

            metrics.update(self.update_discriminator(obs_a, next_obs_a, obs_e, next_obs_e))
            reward = self.compute_reward(obs_a, next_obs_a, reward_a)

        else:
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs = batch

        obs, action, _, discount, next_obs = batch
        
        if self.config.augment:
            batch_size = obs.shape[0]
            
            aug_obs = []
            aug_action = []
            aug_nobs = []
            
            for k in range(batch_size):
                current_ob = obs[k]
                current_action_pick = action[k]
                current_nob = next_obs[k]
                
                new_image, new_action, new_n_image = self.aug.get_data_aug(current_ob, current_action_pick, current_nob)
                
                aug_obs.append(new_image)
                aug_action.append(new_action)
                aug_nobs.append(new_n_image)
                
            aug_obs = np.array(aug_obs)
            aug_action = np.array(aug_action)
            aug_nobs = np.array(aug_nobs)
            
            if self.delta>0:
                aug_action, discount = utils.to_torch((aug_action, discount), self.device)
            else:
                aug_action, reward, discount = utils.to_torch((aug_action, reward, discount), self.device)

        else:
            if self.delta>0:
                aug_action, discount = utils.to_torch((action, discount), self.device)
            else:
                aug_action, reward, discount = utils.to_torch((action, reward, discount), self.device)

            aug_obs = obs
            aug_nobs = next_obs

        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
            
        metrics.update(self.update_critic(aug_obs, aug_action, reward, discount, aug_nobs))
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        
        return metrics
            
            
            
            
                
            
            
            
                
        
        
        
        
        
        