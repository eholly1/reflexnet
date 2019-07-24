from abc import abstractmethod
import itertools
import os
import scipy
import tensorflow as tf

import network


class TFPolicy:

  def __call__(self, obs):
    if isinstance(obs, tf.Tensor):
      return self.forward(obs)
    else:
      obs = tf.convert_to_tensor(obs, dtype=tf.float64)
      res = self.forward(obs)
      return res.eval()

  def save(self, save_path, filename='policy.torch'):
    """Save the policy to a file.
    Args:
      save_path: The path of the dir to save to. May optionally include
        the filename to save to, ending in .torch. If .torch extension
        not included, filename arg will be appended to path.
      filename: The name of the file to save. Default policy.torch.
    """
    raise NotImplementedError

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
    raise NotImplementedError

  @property
  @abstractmethod
  def model(self):
    raise NotImplementedError

  @abstractmethod
  def parameters(self):
    raise NotImplementedError

  def forward(self, obs):
    return self.model(obs)


class FeedForwardPolicy(TFPolicy):

  __scope__ = "feedforward"

  @staticmethod
  def for_env(gym_env, layers_config=[64, 64]):
    obs_size = gym_env.observation_space.shape[0]
    act_size = gym_env.action_space.shape[0]
    return FeedForwardPolicy(obs_size, act_size, layers_config)

  def __init__(self, obs_size, act_size, layers_config=[64, 64]):
    super().__init__()
    with tf.compat.v1.variable_scope(self.__scope__):
      self._model = network.FeedForward(
        input_size=obs_size,
        output_size=act_size,
        layers_config=layers_config)

  def parameters(self):
    return tf.compat.v1.trainable_variables(self.__scope__)

  @property
  def model(self):
    return self._model
