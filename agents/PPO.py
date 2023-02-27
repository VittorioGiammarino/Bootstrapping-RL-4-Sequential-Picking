#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:28:04 2022

@author: vittoriogiammarino
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from models.resnet import ResNet43_8s, MLP
from einops.layers.torch import Rearrange
from torch.distributions.categorical import Categorical

class PPO:
    def __init__(self, config, input_shape, device):
        
        self.device = device
        self.config = config
        
        self.in_shape = input_shape
        max_dim = np.max(self.in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(self.in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
        
        output_channels = self.config.decoder_nc
        
        self.model = ResNet43_8s(self.in_shape[2], output_channels).to(self.device)
        
        self.value_function = MLP(self.in_shape[0]*self.in_shape[1], 1, 
                                  self.config.feature_dim, self.config.hidden_dim).to(self.device)
                           
        #optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.config.learning_rate)
        self.optimizer_value_function = optim.Adam(self.value_function.parameters(), lr = self.config.learning_rate)
        
        self.Total_t = 0
        self.Total_iter = 0
        
    def load_model(self, payload):
        self.model.load_state_dict(payload['agent'].encoder.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.config.learning_rate)
    
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
        
    def forward(self, input_img, softmax_bool):

        in_tens = self.resize_input_img(input_img)
        in_tens = torch.split(in_tens, 1, dim=0)
        logits = ()
        
        for x in in_tens:
            logits += (self.model(x),)
        
        logits = torch.cat(logits, dim=0)
        output = self.pad_rearrange(logits)
        
        if softmax_bool:
            output = torch.log_softmax(torch.clamp(output,-20,20), dim=1)
        
        return output
        
    def forward_value_function(self, input_image):
        pick_score = self.forward(input_image, softmax_bool=False)
        ouput = self.value_function(pick_score)
        return ouput
        
    def act(self, input_image):
        self.eval_mode()
        
        # second version
        pick_conf = self.forward(input_image, softmax_bool=False)
        pi = Categorical(logits = pick_conf)
        
        action = pi.sample()
        action_numpy = action.detach().cpu().numpy()
        pick_conf = pick_conf.reshape(input_image.shape[:2])
        action_reshaped = np.unravel_index(action_numpy, shape=pick_conf.shape)
        picking_pixel = action_reshaped[:2]
        picking_y = int(picking_pixel[0])
        picking_x = int(picking_pixel[1])
                    
        return pi, action, picking_y, picking_x
    
    def GAE(self, done, rewards, values):
        
        number_of_steps = len(done)
        last_value = values[-1]
        last_advantage = 0
        last_return = 0
        
        advantages = np.zeros((number_of_steps,), dtype=np.float32) 
        returns = np.zeros((number_of_steps,), dtype=np.float32) 
        
        for t in reversed(range(number_of_steps)):
            
            mask = 1-done[t]
            last_value = mask*last_value
            last_advantage = mask*last_advantage
            last_return = mask*last_return
            
            delta = rewards[t]+self.config.gae_gamma*last_value-values[t]
            
            last_advantage = delta + self.config.gae_gamma*self.config.gae_lambda*last_advantage
            last_return = rewards[t] + self.config.gae_gamma*last_return
            
            advantages[t] = last_advantage
            returns[t] = last_return
            
            last_value = values[t]
            
        return advantages, returns
    
    def normalize_adv(self, adv):
        return (adv - adv.mean())/(adv.std()+1e-8)
    
    def ppo_loss(self, log_pi, log_pi_old, advantage):
        ratio = torch.exp(log_pi - log_pi_old)
        clipped = torch.clip(ratio, 1-self.config.epsilon, 1+self.config.epsilon)
        policy_loss = torch.minimum(ratio*advantage, clipped*advantage)
        
        return policy_loss, ratio, clipped
    
    def value_loss(self, value, values_old, returns):
        if self.config.TD:
            clipped_value = (value - values_old).clamp(-self.config.epsilon, self.config.epsilon)
            value_loss = torch.max((value-returns)**2,(clipped_value-returns)**2)
        else:
            value_loss = (value-returns)**2
            
        return value_loss
    
    def compute_loss(self, minibatch):
        metrics = dict()
        
        value = self.forward_value_function(minibatch['obs'])
        
        if self.config.TD:
            returns = minibatch['values'] - minibatch['advantages'].reshape(minibatch['values'].shape)
            L_vf = self.value_loss(value, minibatch['values'], returns)
        else:
            returns = minibatch['returns'].reshape(value.shape)
            L_vf = self.value_loss(value, minibatch['values'], returns)
        
        normalize_advantage = self.normalize_adv(minibatch['advantages'])
        
        logits_pick = self.forward(minibatch['obs'], softmax_bool=False)
        pi = Categorical(logits = logits_pick)
        log_pi = pi.log_prob(minibatch['actions'].flatten()).reshape(-1,1)
        
        L_clip, ratio, clipped = self.ppo_loss(log_pi, minibatch['log_pis'], normalize_advantage.reshape(-1,1))
        
        diff_ratio = ratio-clipped
        diff_log_pi = log_pi - minibatch['log_pis']
        
        entropy_bonus = pi.entropy().reshape(-1,1)
        if self.config.entropy:
            loss = (-1)*(L_clip - self.config.c1*L_vf + self.config.c2*entropy_bonus).mean()
        else:
            loss = (-1)*(L_clip - self.config.c1*L_vf).mean()
        
        if self.config.use_tb:
            metrics['returns'] = returns.mean().item()
            metrics['advantage'] = minibatch['advantages'].mean().item()
            metrics['normalized_advantage'] = normalize_advantage.mean().item()
            metrics['actor_log_prob_old'] = minibatch['log_pis'].mean().item()
            metrics['ratio'] = ratio.mean().item() 
            metrics['value'] = value.mean().item()
            metrics['actor_loss'] = L_clip.mean().item()
            metrics['actor_log_prob'] = log_pi.mean().item()
            metrics['actor_ent'] = pi.entropy().mean().item()
            metrics['value_loss'] = L_vf.mean().item()
            metrics['diff_ratio_clipped_ratio'] = diff_ratio.mean().item()
            metrics['diff_log_pi'] = diff_log_pi.mean().item()
        
        return loss, L_clip, L_vf, entropy_bonus, metrics
    
    def train(self, minibatch):
        self.train_mode()
        
        self.optimizer.zero_grad()
        self.optimizer_value_function.zero_grad()
        
        loss, L_clip, L_vf, entropy_bonus, metrics = self.compute_loss(minibatch)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        self.optimizer_value_function.step()
        
        new_loss, new_L_clip, new_L_vf, new_entropy_bonus, _ = self.compute_loss(minibatch)
        
        diff_loss = loss-new_loss
        diff_L_clip = L_clip-new_L_clip
        diff_L_vf = L_vf-new_L_vf
        diff_entropy = entropy_bonus-new_entropy_bonus
        
        if self.config.use_tb:
            metrics['diff_loss_after_backprop'] = diff_loss.mean().item() 
            metrics['diff_L_clip_after_backprop'] = diff_L_clip.mean().item() 
            metrics['diff_L_vf_after_backprop'] = diff_L_vf.mean().item() 
            metrics['diff_entropy_after_backprop'] = diff_entropy.mean().item() 
        
        return metrics
    
    def train_mode(self):
        self.model.train()
        self.value_function.train()

    def eval_mode(self):
        self.model.eval()
        self.value_function.eval()
        
        
        
        
        
        
