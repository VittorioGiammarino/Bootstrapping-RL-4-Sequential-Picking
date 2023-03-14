import torch
from torch import nn
from torch import distributions as torchd

from utils_folder import utils_dreamer as utils

class Random(nn.Module):

  def __init__(self, config, num_actions):
    super(Random, self).__init__()
    self._config = config
    self.num_actions = num_actions

  def actor(self, feat):
    shape = (1, self.num_actions)
    if self._config.actor_dist == 'onehot':
      return utils.OneHotDist(torch.zeros(shape))
    else:
      ones = torch.ones(shape)
      return utils.ContDist(torchd.uniform.Uniform(-ones, ones))

  def train(self, start, context):
    return None, {}


