from abc import abstractmethod
import itertools
import torch

import network


class TorchPolicy(torch.nn.Module):

  def __call__(self, obs):
    if not isinstance(obs, torch.Tensor):
      obs = torch.tensor(obs)
      res = self.forward(obs)
      return res.cpu().detach().numpy()
    else:
      return self.forward(obs)

  @property
  @abstractmethod
  def model(self):
    raise NotImplementedError

  def forward(self, obs):
    return self.model(obs)


class FeedForwardPolicy(TorchPolicy):

  @staticmethod
  def for_env(gym_env, layers_config=[64, 64]):
    obs_size = gym_env.observation_space.shape[0]
    act_size = gym_env.action_space.shape[0]
    return FeedForwardPolicy(obs_size, act_size, layers_config)

  def __init__(self, obs_size, act_size, layers_config=[64, 64]):
    super().__init__()
    self._model = network.FeedForward(
      input_size=obs_size,
      output_size=act_size,
      layers_config=layers_config)

  @property
  def model(self):
    return self._model

# class ReflexPolicy(TorchPolicy):

#   def __init__(self, obs_size, act_size, reflex_layers_config=[5], supervisor_layers_config=[64, 64]):
#     self._obs_size = obs_size
#     self._act_size = act_size
#     self._num_reflexes = obs_size * act_size
#     self._reflexes = [network.FeedForward(input_size=1, output_size=1, layers_config=reflex_layers_config)
#                       for _ in range(self._num_reflexes)]
#     self._reflexes = torch.nn.ModuleList(self._reflexes)
#     self._supervisor = network.FeedForward(
#       input_size=obs_size, output_size=self._num_reflexes, layers_config=supervisor_layers_config)

#   def reflex_parameters(self):
#     return itertools.chain(*[r.parameters() for r in self._reflexes])

#   def supervisor_parameters(self):
#     return self._supervisor.parameters()

#   def forward(self, obs):
#     reflex_activations_raw = self._supervisor(obs)
#     reflex_activations = torch.nn.functional.tanh(reflex_activations_raw)
#     reflex_outputs = []
#     for i in range(self._obs_size):