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
from torch import distributions as torchd

import utils_folder.utils as utils

from models.resnet import Encoder as encoder_net
from models.resnet import Decoder
from einops.layers.torch import Rearrange
from torch.distributions.categorical import Categorical

class Encoder(nn.Module):
    def __init__(self, input_shape, device, from_segm):
        super().__init__()

        self.repr_dim = 512*20*20
        
        self.device = device
        self.from_segm = from_segm
        
        self.in_shape = input_shape
        max_dim = np.max(self.in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(self.in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        self.net = encoder_net(self.in_shape[2])
        self.apply(utils.init_xavier_weights)

    def forward(self, input_img):
        if len(input_img.shape)==3:
            input_img = self.resize_input_img(input_img)
        elif len(input_img.shape)==4:
            input_img = self.process_img(input_img)

        in_tens = torch.split(input_img, 1, dim=0)
        h = ()
        
        for x in in_tens:
            h += (self.net(x),)

        h = torch.cat(h, dim=0)

        return h
        
    def get_feature_vector(self, h):
        return h.reshape(h.shape[0], -1)

    def process_img(self, input_img):
        img = self.normalize(input_img)
        return img
        
    def normalize(self, input_img):
        img = (input_img / 255) - 0.5
        return img
    
    def resize_input_img(self, input_img):
        in_data = np.pad(input_img, self.padding, mode='constant')
        in_data_processed = self.process_img(in_data)
        in_shape = (1,) + in_data_processed.shape
        in_data_processed = in_data_processed.reshape(in_shape)
        in_tens = torch.tensor(in_data_processed, dtype=torch.float32).to(self.device)
            
        return in_tens  

    def resize_segm(self, input_img):
        in_data_processed = np.pad(input_img, self.padding, mode='constant')
        in_shape = (1,) + in_data_processed.shape
        in_data_processed = in_data_processed.reshape(in_shape)
        in_tens = torch.tensor(in_data_processed, dtype=torch.float32).to(self.device)
            
        return in_tens  
    
class Critic(nn.Module):
    def __init__(self, input_shape, output_channels):
        super().__init__()

        self.in_shape = input_shape
        max_dim = np.max(self.in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(self.in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
        
        self.critic = Decoder(output_channels)
        self.apply(utils.init_xavier_weights)
        
    def forward(self, h):
        logits_critic = self.critic(h)
        output_critic = self.pad_rearrange(logits_critic)
        
        return output_critic
            
    def pad_rearrange(self, logits):
        c0 = self.padding[:2, 0]
        c1 = c0 + self.in_shape[:2]
        logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

        output = Rearrange('b h w c -> b (h w c)')(logits)
        
        return output
        
class REDQ_Agent:
    def __init__(self, input_shape, workspace, device, use_tb, critic_target_tau, decoder_nc, learning_rate, 
                num_Q, num_min, num_update, exploration_rate, num_expl_steps, from_segm=False,
                safety_mask=False):
        
        self.device = device
        self.use_tb = use_tb
        self.critic_target_tau = critic_target_tau
        self.from_segm = from_segm
        self.num_Q = num_Q
        self.num_min = num_min
        self.num_update = num_update

        self.workspace = workspace
        self.exploration_rate = exploration_rate
        self.num_expl_steps = num_expl_steps
        self.safety_mask = safety_mask
        
        output_channels = decoder_nc
        self.encoder = Encoder(input_shape, device, from_segm).to(self.device)
        print("Training using only exogenous reward")

        self.critic_list, self.critic_target_list = [], []
        for _ in range(self.num_Q):
            Q_net = Critic(input_shape, output_channels).to(self.device)
            self.critic_list.append(Q_net)
            Q_net_target = Critic(input_shape, output_channels).to(self.device)
            Q_net_target.load_state_dict(Q_net.state_dict())
            Q_net_target.train()
            self.critic_target_list.append(Q_net_target)
        
        #optimizers
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.optimizer_critic_list = []
        for Q in range(self.num_Q):
            self.optimizer_critic_list.append(optim.Adam(self.critic_list[Q].parameters(), lr=learning_rate))
        
        self.train()
        
    def train(self, training=True):
        self.training = training 
        for q in range(self.num_Q):
            self.critic_list[q].train(training)

    def check_in_workspace(self, cartesian_position_box):
        in_workspace = False
        x = cartesian_position_box[0]
        y = cartesian_position_box[1]
        z = cartesian_position_box[2]
        
        if x>=self.workspace[0][0] and x<=self.workspace[0][1]:
            if y>=self.workspace[1][0] and y<=self.workspace[1][1]:
                if z>=self.workspace[2][0] and z<=self.workspace[2][1]:
                    in_workspace = True
                    
        return in_workspace

    def compute_safety_mask(self, target_V, xyz, input_image_shape):

        num_pixels = target_V.shape[1]
        mask = torch.ones([1, num_pixels], dtype=torch.float).to(self.device)
        valid = torch.zeros([1, num_pixels], dtype=torch.int).to(self.device)

        for i in range(num_pixels):
            matrix_pixels = target_V.reshape(input_image_shape)
            index_reshaped = np.unravel_index(i, shape=matrix_pixels.shape)
            pick = index_reshaped[:2]
            pick_y = int(pick[0])
            pick_x = int(pick[1]) 
            pick_position = xyz[pick_y, pick_x]

            in_workspace = self.check_in_workspace(pick_position)

            if in_workspace:
                valid[0,i] = 1
                mask[0,i] = +100
            else:
                mask[0,i] = -100

        return mask, valid
        
    def act(self, input_image, xyz, step, eval_mode=True):

        obs = self.encoder(input_image)
        input_image_shape = input_image.shape[:2]
        target_Q_list = []
        for i in range(self.num_Q):
            target_Q = self.critic_target_list[i](obs).reshape(-1,1)
            target_Q_list.append(target_Q)

        target_Q_cat = torch.cat(target_Q_list,-1)
        target_V, _ = torch.min(target_Q_cat, dim=-1)
        target_V = target_V.reshape(1,-1)
                    
        if eval_mode:
            action, picking_y, picking_x = self.act_eval(target_V, xyz, input_image_shape)

        elif step <= self.num_expl_steps:
            action, picking_y, picking_x = self.explore(target_V, xyz, input_image_shape)
            
        else:
            action, picking_y, picking_x = self.act_training(target_V, xyz, input_image_shape)
            
        return action, picking_y, picking_x

    def act_eval(self, target_V, xyz, input_image_shape):

        if self.safety_mask:
            mask, _ = self.compute_safety_mask(target_V, xyz, input_image_shape)
            pick_conf = target_V + mask

        else:
            pick_conf = target_V

        pick_conf = pick_conf.detach().cpu().numpy()
        pick_conf = np.float32(pick_conf).reshape(input_image_shape)

        #this one works cause we are processing a single image at the time during eval mode        
        action = np.argmax(pick_conf) 
        action = np.unravel_index(action, shape=pick_conf.shape)
        
        picking_pixel = action[:2]
        picking_y = picking_pixel[0]
        picking_x = picking_pixel[1]   

        return action, picking_y, picking_x

    def explore(self, target_V, xyz, input_image_shape):
        print("Explore")

        if self.safety_mask:
            _, valid = self.compute_safety_mask(target_V, xyz, input_image_shape)
            try:
                valid = valid.detach().cpu().numpy()
                action_numpy = np.random.choice(valid.nonzero()[1], size=1)
            except:
                num_pixels = target_V.shape[1]
                action_numpy = np.random.randint(num_pixels, size=(1,))

        else:
            num_pixels = target_V.shape[1]
            action_numpy = np.random.randint(num_pixels, size=(1,))

        action_reshaped = np.unravel_index(action_numpy, shape=input_image_shape)
        
        picking_pixel = action_reshaped[:2]
        picking_y = int(picking_pixel[0])
        picking_x = int(picking_pixel[1])

        action = action_numpy

        return action, picking_y, picking_x

    def act_training(self, target_V, xyz, input_image_shape):

        pick_conf = torch.clone(target_V)
        expl_rv = np.random.rand()

        if self.safety_mask:
            mask, valid = self.compute_safety_mask(target_V, xyz, input_image_shape)

            if expl_rv <= self.exploration_rate:
                print("Explore")
                try:
                    valid = valid.detach().cpu().numpy()
                    action_numpy = np.random.choice(valid.nonzero()[1], size=1)
                except:
                    num_pixels = target_V.shape[1]
                    action_numpy = np.random.randint(num_pixels, size=(1,))

            else:
                pick_conf = target_V + mask
                pi = Categorical(logits = pick_conf)
                action = pi.sample()
                action_numpy = action.detach().cpu().numpy()
        else:

            if expl_rv <= self.exploration_rate:
                print("Explore")
                num_pixels = target_V.shape[1]
                action_numpy = np.random.randint(num_pixels, size=(1,))

            else:
                pi = Categorical(logits=pick_conf)
                action = pi.sample()
                action_numpy = action.detach().cpu().numpy()

        action_reshaped = np.unravel_index(action_numpy, shape=input_image_shape)
        
        picking_pixel = action_reshaped[:2]
        picking_y = int(picking_pixel[0])
        picking_x = int(picking_pixel[1])

        action = action_numpy

        return action, picking_y, picking_x
    
    def act_batch(self, input_image, sample_idxs):   

        target_Q_list = []
        for i in sample_idxs:
            target_Q = self.critic_target_list[i](input_image)
            target_Q = target_Q.reshape(target_Q.shape + (1,))
            target_Q_list.append(target_Q)

        target_Q_cat = torch.cat(target_Q_list, -1)
        target_V, _ = torch.min(target_Q_cat, dim=-1)

        pick_conf = target_V
        argmax = torch.argmax(pick_conf, dim=-1)
        pi = Categorical(logits = pick_conf)
            
        return target_V, argmax, pi
        
    def update_critic(self, obs, action, reward, discount, next_obs):
        metrics = dict()

        sample_idxs = np.random.choice(self.num_Q, self.num_min, replace=False)
        
        with torch.no_grad():
            target_V, next_pick, pi = self.act_batch(next_obs, sample_idxs)
            target_V_pick = target_V.gather(1, next_pick.reshape(-1,1))
            target_Q = reward + (discount*target_V_pick)

        pick = action[:,0]
        critic_loss = 0

        for i in range(self.num_Q):
            Q = self.critic_list[i](obs)
            Q_picked = Q.gather(1, pick.reshape(-1,1).long())
            critic_loss = critic_loss + F.mse_loss(Q_picked, target_Q)

        critic_loss = critic_loss*self.num_Q
                        
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['policy_entropy'] = pi.entropy().mean().item()
                        
        self.optimizer_encoder.zero_grad(set_to_none=True)
        for i in range(self.num_Q):
            self.optimizer_critic_list[i].zero_grad(set_to_none=True)

        critic_loss.backward()

        for i in range(self.num_Q):
            self.optimizer_critic_list[i].step()

        self.optimizer_encoder.step()
        
        return metrics
        
    def update(self, replay_iter, step):
        metrics = dict()
        
        for i_update in range(self.num_update):

            batch = next(replay_iter)
            obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
            
            if self.use_tb:
                metrics['batch_reward'] = reward.mean().item()

            obs = obs.float()
            next_obs = next_obs.float()    

            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)
                
            metrics.update(self.update_critic(obs, action, reward, discount, next_obs))

            for i in range(self.num_Q):
                utils.soft_update_params(self.critic_list[i], self.critic_target_list[i], self.critic_target_tau)
        
        return metrics
            
            
            
            
                
            
            
            
                
        
        
        
        
        
        