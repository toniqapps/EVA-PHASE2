# -*- coding: utf-8 -*-
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from collections import deque


def conv2d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1))

def transition_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(0.1))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=state_dim, out_channels=16, kernel_size=(3, 3), padding=0), 
            nn.ReLU()) 
        self.convblock2 = conv2d_block(in_channels=16, out_channels=10)
        self.transitionblock = transition_block(in_channels=10, out_channels=10)
        self.convblock3 = conv2d_block(in_channels=10, out_channels=10)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )
        self.fc1 = nn.Linear(12, 10)
        self.fc2 = nn.Linear(10, 1)
        self.max_action = max_action

    def forward(self, x, y):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = torch.cat([x, y], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.max_action * torch.tanh(x)
        return x


class TD3(object):
  
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.max_action = max_action

    def select_action(self, state, orientation):
        state = torch.Tensor(state)
        state = state.unsqueeze(0).unsqueeze(0)
        orientation = torch.Tensor(orientation.reshape(1, -1))
        return self.actor(state, orientation).cpu().data.numpy().flatten()

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        #torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=torch.device('cpu')))
        #self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


      

