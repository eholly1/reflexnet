from abc import abstractmethod
import itertools
import os
import scipy
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


class MetricPolicy(TorchPolicy):

  def __init__(self, obs_size, embedding_size, data, layers_config=[64, 64], k=1):
    super().__init__()
    self._model = network.FeedForward(
      input_size=obs_size,
      output_size=embedding_size,
    )
    self._data = data
    self._kdtree = None
    self._k = k

  def get_embedding(self, obs):
    return self._model(obs)

  def rebuild(self):
    embeddings = self.get_embedding(data["obs"])
    self._kdtree = scipy.spatial.KDTree(embeddings)

  def forward(self, obs):
    embedding = self.get_embedding(obs)
    neighbor_dists, neighbor_idxs = self._kdtree.query(embedding, k)
    action = self._data[neighbor_idxs[0]]


class ReflexPolicy(TorchPolicy):

  def __init__(
    self,
    obs_size,
    act_size,
    num_latent_dims=25,
    reflex_layers_config=[5],
    supervisor_layers_config=[64, 64]):

    self._obs_size = obs_size
    self._act_size = act_size
    self._num_latent_dims = num_latent_dims

    self._reflex_modules = []  # For constructing a ModuleList.

    # self._reflexes is a 2d-array of reflexes, from obs index to action index.
    self._reflexes = []
    for _ in range(self._obs_size):
      reflex_act_list = []
      self._reflexes.append(reflex_act_list)
      for _ in range(self._act_size):
        reflex = network.FeedForward(
          input_size=1, output_size=1, layers_config=relfex_layers_config)
        reflex_act_list.append(reflex)
        self._reflex_modules.append(reflex)
    self._reflex_modules = torch.nn.ModuleList(self._reflex_modules)

    # The supervisor predicts latents 
    self._supervisor = network.FeedForward(
      input_size=obs_size, output_size=self._num_latent_dims,
      layers_config=supervisor_layers_config, output_activation=torch.nn.Sigmoid)

    self._activation_matrix = torch.nn.Parameter(
      torch.ones(self._num_latent_dims, self._obs_size, self._act_size))
    self._activation_softmax = torch.nn.Softmax(dim=-1)

  def reflex_parameters(self):
    return itertools.chain(*[r.parameters() for r in self._reflex_modules])

  def supervisor_parameters(self):
    raise NotImplementedError # Supervisor and activation parameters.

  def forward(self, obs):
    import pdb
    pdb.set_trace()

    # Use supervisor to modulate latents.
    latent_activations = self._supervisor(obs)

    # Transform latents into activations over reflexes.
    reflex_activation_logits = torch.mm(latent_activations, self._activation_matrix)
    reflex_activations = self._activation_softmax(reflex_activation_logits)

    # Compute outputs of all reflexes.
    reflex_outputs = []
    for io in range(self._obs_size):
      reflex_act_outputs = []
      for ia in range(self._act_size):
        reflex_act_outputs.append(self._reflexes[io][ia](obs[:, io]))
      reflex_act_outputs = torch.stack(reflex_act_outputs)
      reflex_outputs.append(reflex_act_outputs)
    reflex_outputs = torch.stack(reflex_act_outputs)

    # Weight the reflexes by activations, then sum along obs dim.
    reflex_values = reflex_outputs * reflex_activations
    reflex_values = torch.sum(reflex_values, dim=1)

    return reflex_values
    
