#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:11:14 2022

@author: vittoriogiammarino
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from models.resnet import ResNet43_8s
from einops.layers.torch import Rearrange

class Learner:
    def __init__(self, config, input_shape, device):
        
        self.device = device
        self.config = config
        
        self.in_shape = input_shape
        max_dim = np.max(self.in_shape[:2])
        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(self.in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)
                
        encoder_output_channels = 1 
        
        self.encoder = ResNet43_8s(self.in_shape[2], encoder_output_channels).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=self.config.learning_rate)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        
    def process_img(self, input_img):
        img = self.normalize(input_img)
        return img
    
    def normalize(self, input_img):
        img = (input_img / 255)
        return img
    
    def resize_input_img(self, input_img):
        in_data = np.pad(input_img, self.padding, mode='constant')
        in_data_processed = self.process_img(in_data)
        in_shape = (1,) + in_data_processed.shape
        in_data_processed = in_data_processed.reshape(in_shape)
        in_tens = torch.tensor(in_data_processed, dtype=torch.float32).to(self.device)
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
            logits += (self.encoder(x),)
        
        logits = torch.cat(logits, dim=0)
        output = self.pad_rearrange(logits)
        
        if softmax_bool:
            output = nn.Softmax(dim=1)(output)
            output = output.detach().cpu().numpy()
            output = np.float32(output).reshape(logits.shape[1:])
        
        return output
            
    def act(self, input_image):
        self.eval_mode()
        pick_conf = self.forward(input_image, softmax_bool=True)
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        picking_pixel = argmax[:2]
        picking_y = picking_pixel[0]
        picking_x = picking_pixel[1]
        return picking_y, picking_x
                
    def get_picking_label(self, pixel_label, input_img):
        label_size = input_img.shape[:2] + (1,)
        label = np.zeros(label_size)
        label[pixel_label[0][0], pixel_label[0][1], 0] = 1
        label = torch.tensor(label, dtype=torch.float32).to(self.device)
        label_flatten = Rearrange('h w c -> 1 (h w c)')(label)
        label = torch.argmax(label_flatten, dim=1)
        return label, label_flatten
    
    def compute_loss(self, input_image, action_label):
        output = self.forward(input_image, softmax_bool=False)
        label, label_flatten = self.get_picking_label(action_label, input_image)
        loss = self.loss(output, label) 
        return loss
    
    def train(self, input_image, action_label):
        """Train."""
        self.train_mode()
        self.optimizer.zero_grad()
        loss = self.compute_loss(input_image, action_label)
        loss.backward()
        self.optimizer.step()
        return np.float32(loss.detach().cpu().numpy())
    
    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval() 
        
        
        
        
        
        
