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
    def __init__(self, input_shape, device):
        super().__init__()

        self.repr_dim = 512*20*20
        
        self.device = device
        
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
    
class Critic(nn.Module):
    def __init__(self, input_shape, output_channels):
        super().__init__()

        self.in_shape = input_shape
        max_dim = np.max(self.in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(self.in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
        
        self.critic_1 = Decoder(output_channels)
        self.critic_2 = Decoder(output_channels)
        self.apply(utils.init_xavier_weights)
        
    def forward(self, h):
        
        logits_critic_1 = self.critic_1(h)
        logits_critic_2 = self.critic_2(h)
        
        output_critic_1 = self.pad_rearrange(logits_critic_1)
        output_critic_2 = self.pad_rearrange(logits_critic_2)
        
        return output_critic_1, output_critic_2
            
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
        
class DrQAgent_adv:
    def __init__(self, input_shape, device, use_tb, critic_target_tau, update_every_steps, decoder_nc, learning_rate, 
                reward_d_coef, imitation_learning, RL, learning_rate_discriminator, feature_dim, hidden_dim, augmentation,
                GAN_loss='bce'):
        
        self.device = device
        self.use_tb = use_tb
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.GAN_loss = GAN_loss
        
        output_channels = decoder_nc

        self.encoder = Encoder(input_shape, device).to(self.device)


        if self.GAN_loss == 'least-square':
            self.discriminator = Discriminator(2*self.encoder.repr_dim, 2*feature_dim, 2*feature_dim, hidden_dim).to(device)
            self.reward_d_coef = reward_d_coef

        elif self.GAN_loss == 'bce':
            self.discriminator = Discriminator(2*self.encoder.repr_dim, 2*feature_dim, 2*feature_dim, hidden_dim, dist='binary').to(device)
        else:
            NotImplementedError
        
        self.imitation_learning = imitation_learning
        self.RL = RL

        if self.imitation_learning:
            print("Training only by imitation")
        elif self.RL:
            print("Training using only exogenous reward")
        else:
            print("Training using both imitation and exogenous reward")
        
        self.critic = Critic(input_shape, output_channels).to(self.device)
        self.critic_target = Critic(input_shape, output_channels).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        #optimizers
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.discriminator_opt = optim.Adam(self.discriminator.parameters(), lr=learning_rate_discriminator)
        
        # data augmentation
        self.augmentation = augmentation
        self.aug = RandomShiftsAug(pad=8)
        
        self.train()
        self.critic_target.train()
        
    def train(self, training=True):
        self.training = training 
        self.critic.train(training)
        self.discriminator.train(training)
        
    def act(self, input_image, eval_mode=True):

        obs = self.encoder(input_image)
                    
        if eval_mode:
            target_Q1, target_Q2 = self.critic_target(obs)         
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
            
            target_Q1, target_Q2 = self.critic_target(obs)
            target_V = torch.min(target_Q1, target_Q2) 
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
    
    def act_batch(self, input_image):        
        target_Q1, target_Q2 = self.critic_target(input_image)
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
            
        Q1, Q2 = self.critic.forward(obs)
        
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
                        
        self.optimizer_encoder.zero_grad(set_to_none=True)
        self.optimizer_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.optimizer_critic.step()
        self.optimizer_encoder.step()
        
        return metrics

    def compute_reward(self, obs_a, next_a):
        metrics = dict()

        # augment
        if self.augmentation:
            obs_a = self.aug(obs_a.float())
            next_a = self.aug(next_a.float())
        else:
            obs_a = obs_a.float()
            next_a = next_a.float()

        # encode
        with torch.no_grad():
            obs_a = self.encoder(obs_a)
            next_a = self.encoder(next_a)

            self.discriminator.eval()

            h = self.encoder.get_feature_vector(obs_a)
            next_h = self.encoder.get_feature_vector(next_a)

            transition_h = torch.cat([h, next_h], dim = -1)
            d = self.discriminator(transition_h)

            if self.GAN_loss == 'least-square':
                reward_d = self.reward_d_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            else:
                reward_d = d.mode()
            
            reward = reward_d 

            if self.use_tb:
                metrics['reward_d'] = reward_d.mean().item()
    
            self.discriminator.train()
            
        return reward, metrics
    
    def compute_discriminator_grad_penalty_LS(self, obs_e, next_obs_e, lambda_=10):

        expert_data = torch.cat([obs_e, next_obs_e], dim=-1)
        expert_data.requires_grad = True
        
        d = self.discriminator(expert_data)
        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=expert_data, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        
        return grad_pen

    def compute_discriminator_grad_penalty_bce(self, obs_a, next_a, obs_e, next_e, lambda_=10):

        agent_feat = torch.cat([obs_a, next_a], dim=-1)
        alpha = torch.rand(agent_feat.shape[:1]).unsqueeze(-1).to(self.device)
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        disc_penalty_input = alpha*agent_feat + (1-alpha)*expert_data

        disc_penalty_input.requires_grad = True

        d = self.discriminator(disc_penalty_input).mode()

        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=disc_penalty_input, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
        
    def update_discriminator(self, obs_a, next_a, obs_e, next_e):
        metrics = dict()

        h_a = self.encoder.get_feature_vector(obs_a)
        next_h_a = self.encoder.get_feature_vector(next_a)

        h_e = self.encoder.get_feature_vector(obs_e)
        next_h_e = self.encoder.get_feature_vector(next_e)

        transition_a = torch.cat([h_a, next_h_a], dim=-1)
        transition_e = torch.cat([h_e, next_h_e], dim=-1)
        
        agent_d = self.discriminator(transition_a)
        expert_d = self.discriminator(transition_e)

        if self.GAN_loss == 'least-square':
            expert_labels = 1.0
            agent_labels = -1.0
            
            expert_loss = F.mse_loss(expert_d, expert_labels*torch.ones(expert_d.size(), device=self.device))
            agent_loss = F.mse_loss(agent_d, agent_labels*torch.ones(agent_d.size(), device=self.device))
            grad_pen_loss = self.compute_discriminator_grad_penalty_LS(h_e.detach(), next_h_e.detach())
            
            loss = 0.5*(expert_loss + agent_loss) + grad_pen_loss

        elif self.GAN_loss == 'bce':
            expert_loss = (expert_d.log_prob(torch.ones_like(expert_d.mode()).to(self.device))).mean()
            agent_loss = (agent_d.log_prob(torch.zeros_like(agent_d.mode()).to(self.device))).mean()
            grad_pen_loss = self.compute_discriminator_grad_penalty_bce(h_a.detach(), next_h_a.detach(), h_e.detach(), next_h_e.detach())
            loss = -(expert_loss+agent_loss) + grad_pen_loss
        
        # optimize inverse models
        self.discriminator_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_opt.step()
        
        if self.use_tb:
            metrics['discriminator_expert_loss'] = expert_loss.item()
            metrics['discriminator_agent_loss'] = agent_loss.item()
            metrics['discriminator_loss'] = loss.item()
            metrics['discriminator_grad_pen'] = grad_pen_loss.item()
            
        return metrics   
        
    def update(self, replay_iter, replay_buffer_expert, step):
        metrics = dict()
        
        if step % self.update_every_steps != 0:
            return metrics

        batch_expert = next(replay_buffer_expert)
        obs_e_raw, _, next_obs_e_raw = utils.to_torch(batch_expert, self.device)

        batch = next(replay_iter)
        obs, action, reward_a, discount, next_obs = utils.to_torch(batch, self.device)

        if self.augmentation:
            obs_e = self.aug(obs_e_raw.float())
            next_obs_e = self.aug(next_obs_e_raw.float())
            obs_a = self.aug(obs.float())
            next_obs_a = self.aug(next_obs.float())
        else:
            obs_e = obs_e_raw.float()
            next_obs_e = next_obs_e_raw.float()
            obs_a = obs.float()
            next_obs_a = next_obs.float()

        with torch.no_grad():
            obs_e = self.encoder(obs_e)
            next_obs_e = self.encoder(next_obs_e)
            obs_a = self.encoder(obs_a)
            next_obs_a = self.encoder(next_obs_a)

        metrics.update(self.update_discriminator(obs_a, next_obs_a, obs_e, next_obs_e))
        reward_d, metrics_r = self.compute_reward(obs, next_obs)
        metrics.update(metrics_r)

        if self.imitation_learning:
            reward = reward_d
        elif self.RL:
            reward = reward_a
        else:
            reward = reward_d + reward_a
        
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

        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        
        return metrics
            
            
            
            
                
            
            
            
                
        
        
        
        
        
        