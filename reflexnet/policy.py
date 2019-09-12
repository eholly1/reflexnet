from abc import abstractmethod
import itertools
import numpy as np
import os
import scipy
import torch

import network


class TorchPolicy(torch.nn.Module):

  @classmethod
  def for_env(cls, gym_env, *args, **kwargs):
    obs_size = gym_env.observation_space.shape[0]
    act_size = gym_env.action_space.shape[0]
    return cls(
      obs_size, act_size, *args, **kwargs)

  def __call__(self, obs):
    if not isinstance(obs, torch.Tensor):
      obs = torch.tensor(obs)
      res = self.forward(obs)
      return res.cpu().detach().numpy()
    else:
      return self.forward(obs)

  def save(self, save_path, filename='policy.torch'):
    """Save the policy to a file.
    Args:
      save_path: The path of the dir to save to. May optionally include
        the filename to save to, ending in .torch. If .torch extension
        not included, filename arg will be appended to path.
      filename: The name of the file to save. Default policy.torch.
    """
    if save_path.endswith('.torch'):
      split_path = save_path.split('/')
      filename = split_path[-1]
      save_path = os.path.join(split_path[:-1])
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    with open(os.path.join(save_path, filename), 'wb') as f:
      torch.save(self, f)

  @staticmethod
  def load(load_path, filename=None):
    """Load a policy from a file.
    Args:
      load_path: The path to the policy to load. May optionally
        be a path to a directory containing policies.
      filename: The filename of the dir to load. If not specified
        and load_path is a dir, will load the latest policy from
        within that dir.
    Returns:
      The loaded policy.
    """
    if not load_path.endswith('.torch'):
      if filename is None:
        # Get the latest filename.
        all_filenames = [
          fname for fname in os.listdir(load_path)
          if fname.endswith('.torch')
          ]
        mtime_to_fname = {
          os.path.getmtime(os.path.join(load_path, fname)): fname
          for fname in all_filenames
          }
        latest_mtime = sorted(mtime_to_fname.keys())[-1]
        filename = mtime_to_fname[latest_mtime]
      load_path = os.path.join(load_path, filename)
    return torch.load(load_path)

  @property
  @abstractmethod
  def model(self):
    raise NotImplementedError

  def forward(self, obs):
    return self.model(obs)


class FeedForwardPolicy(TorchPolicy):

  def __init__(self, obs_size, act_size, layers_config=[64, 64]):
    super().__init__()
    self._model = network.FeedForward(
      input_size=obs_size,
      output_size=act_size,
      layers_config=layers_config)

  @property
  def model(self):
    return self._model


class HRMPolicy(TorchPolicy):

  def __init__(self, obs_size, act_size):
    super().__init__()
    self._model = network.HRMNet(
      input_size=obs_size,
      output_size=act_size,
    )

  def reset(self):
    self._model.reset()

  @property
  def model(self):
    return self._model


class SoftKNNPolicy(TorchPolicy):

  def __init__(self, obs_size, act_size, layers_config=[64, 64]):
    super().__init__()
    self._model = network.SoftKNN(
      input_size=obs_size,
      output_size=act_size)

  @property
  def model(self):
    return self._model


class ReflexPolicy(TorchPolicy):

  @staticmethod
  def for_env(gym_env, num_reflexes=25, ref_layers_config=[16], sup_layers_config=[32]):
    obs_size = gym_env.observation_space.shape[0]
    act_size = gym_env.action_space.shape[0]
    return ReflexPolicy(
      obs_size, act_size, num_reflexes=num_reflexes,
      ref_layers_config=ref_layers_config, sup_layers_config=sup_layers_config)

  def __init__(self, obs_size, act_size, num_reflexes=25, ref_layers_config=[16], sup_layers_config=[32, 32]):
    super().__init__()
    self._obs_size = obs_size
    self._act_size = act_size
    self._num_reflexes = num_reflexes

    # Make reflex subnetworks. For each action dimension, there is a subnetwork 
    self._reflexes = []
    for a in range(act_size):
      action_reflexes = []
      for r in range(self._num_reflexes):
        action_reflexes.append(network.FeedForward(obs_size, 1, layers_config=ref_layers_config))
      action_reflexes = torch.nn.ModuleList(action_reflexes)
      self._reflexes.append(action_reflexes)
    self._reflexes = torch.nn.ModuleList(self._reflexes)

    # Network for selecting amongst reflexes, given observation.
    self._supervisor = network.FeedForward(obs_size, act_size * num_reflexes, layers_config=sup_layers_config)
    self._softmax = torch.nn.Softmax(dim=-1)

  def parameters(self):
    return self._supervisor.parameters(), self._reflexes.parameters()

  def reflex_softmax_weights(self, obs):
    if len(obs.shape) == 1:
      reflex_logits = self._supervisor(obs).view(self._act_size, self._num_reflexes)
    elif len(obs.shape) == 2:
      reflex_logits = self._supervisor(obs).view(-1, self._act_size, self._num_reflexes)
    else:
      raise ValueError('ReflexPolicy currently only supports observations with one or no batch dimensions')

    return self._softmax(reflex_logits)  # Softmax over reflex dimension.

  def reflex_outputs(self, obs):
    reflex_outputs = []
    for r in range(self._num_reflexes):
      action_outputs = []
      for a in range(self._act_size):
        action_outputs.append(self._reflexes[a][r](obs))
      action_outputs = torch.cat(action_outputs, dim=-1)
      reflex_outputs.append(action_outputs)
    reflex_outputs = torch.stack(reflex_outputs, dim=-1)
    return reflex_outputs

  def forward(self, obs):
    reflex_outputs = self.reflex_outputs(obs)
    reflex_softmax_weights = self.reflex_softmax_weights(obs)
    weighted_reflex_outputs =  reflex_outputs * reflex_softmax_weights
    action_outputs = torch.sum(weighted_reflex_outputs, dim=-1)
    return action_outputs


# class MetricPolicy(TorchPolicy):

#   def __init__(self, obs_size, embedding_size, data, layers_config=[64, 64], k=1):
#     super().__init__()
#     self._model = network.FeedForward(
#       input_size=obs_size,
#       output_size=embedding_size,
#     )
#     self._data = data
#     self._kdtree = None
#     self._k = k

#   def get_embedding(self, obs):
#     return self._model(obs)

#   def rebuild(self):
#     embeddings = self.get_embedding(data["obs"])
#     self._kdtree = scipy.spatial.KDTree(embeddings)

#   def forward(self, obs):
#     embedding = self.get_embedding(obs)
#     neighbor_dists, neighbor_idxs = self._kdtree.query(embedding, k)
#     action = self._data[neighbor_idxs[0]]
