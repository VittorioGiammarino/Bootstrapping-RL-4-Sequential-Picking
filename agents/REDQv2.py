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
from models.resnet import Decoder, MLP, Discriminator
from einops.layers.torch import Rearrange
from torch.distributions.categorical import Categorical
from utils_folder.utils import Bernoulli

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = Rearrange('b h w c -> b c h w')(x)
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

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

        if self.from_segm:
            if len(input_img.shape)==3:
                input_img = self.resize_segm(input_img)
            elif len(input_img.shape)==4:
                input_img = self.process_img(input_img)
            
        else:
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

class Discriminator(nn.Module):
    def __init__(self, repr_dim, feature_dim, input_net_dim, hidden_dim, dist=None):
        super().__init__()
                
        self.dist = dist
        self._shape = (1,)
        self.repr_dim = repr_dim
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.net = nn.Sequential(nn.Linear(input_net_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 1))  
        
        self.apply(utils.weight_init)
       
    def forward(self, transition):
        d = self.net(self.trunk(transition))

        if self.dist == 'binary':
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=d), len(self._shape)))
        else:
            return d 
        
class REDQAgent:
    def __init__(self, input_shape, device, use_tb, critic_target_tau, decoder_nc, learning_rate, 
                num_Q, num_min, num_update, augmentation, from_segm=False):
        
        self.device = device
        self.use_tb = use_tb
        self.critic_target_tau = critic_target_tau
        self.from_segm = from_segm
        self.num_Q = num_Q
        self.num_min = num_min
        self.num_update = num_update
        
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

        # data augmentation
        self.augmentation = augmentation
        self.aug = RandomShiftsAug(pad=8)
        
        self.train()
        
    def train(self, training=True):
        self.training = training 
        for q in range(self.num_Q):
            self.critic_list[q].train(training)
        
    def act(self, input_image, eval_mode=True):

        obs = self.encoder(input_image)
                    
        if eval_mode:

            target_Q_list = []
            for i in range(self.num_Q):
                target_Q = self.critic_target_list[i](obs).reshape(-1,1)
                target_Q_list.append(target_Q)

            target_Q_cat = torch.cat(target_Q_list,-1)
            target_V, _ = torch.min(target_Q_cat, dim=-1)
            target_V = target_V.reshape(1,-1)
            
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
            
            target_Q_list = []
            for i in range(self.num_Q):
                target_Q = self.critic_target_list[i](obs).reshape(-1,1)
                target_Q_list.append(target_Q)

            target_Q_cat = torch.cat(target_Q_list,-1)
            target_V, _ = torch.min(target_Q_cat, dim=-1)
            target_V = target_V.reshape(1,-1)

            pick_conf = target_V
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
        
    def update(self, replay_iter, replay_buffer_expert, step):
        metrics = dict()
        
        for i_update in range(self.num_update):
            batch_expert = next(replay_buffer_expert)
            obs_e_raw, _, next_obs_e_raw = utils.to_torch(batch_expert, self.device)

            batch = next(replay_iter)
            obs, action, reward_a, discount, next_obs = utils.to_torch(batch, self.device)

            reward = reward_a
            
            if self.use_tb:
                metrics['batch_reward'] = reward_a.mean().item()

            # augment
            if self.augmentation:
                obs = self.aug(obs.float())
                next_obs = self.aug(next_obs.float())
            else:
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
            
            
            
            
                
            
            
            
                
        
        
        
        
        
        