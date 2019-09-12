import numpy as np
import os
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

import policy
import summaries
import trainer
import utils

# TODO(eholly): Make generic torch in-memory Dataset class.
class BCFrameDataset(trainer.Dataset):

  def __init__(self, batch_size, load_path, eval_fraction=0.2, max_size=200000):
    self._batch_size = batch_size
    self._max_size = max_size

    if os.path.isfile(load_path):
      rollout_data = torch.load(load_path)
    elif os.path.isdir(load_path):
      rollout_data_list = utils.load_subdirs(load_path)
      rollout_data = utils.tree_apply(utils.merge_packed_sequences, *rollout_data_list)
    else:
      raise ValueError('Load path not found: %s' % load_path)
    self._init_dataset_with_rollout_data(rollout_data)
    self._eval_fraction = eval_fraction
    self._update_train_cutoff()
    
  def _update_train_cutoff(self):
    self._train_cutoff = int(self.N * (1.0 - self._eval_fraction))

  @property
  def N(self):
    return self._N

  def _init_dataset_with_rollout_data(self, rollout_data):
    assert isinstance(rollout_data['obs'], torch.nn.utils.rnn.PackedSequence)
    assert isinstance(rollout_data['act'], torch.nn.utils.rnn.PackedSequence)

    # Store states and actions with single batch dim.
    self._obs, self._ep_lens = torch.nn.utils.rnn.pad_packed_sequence(rollout_data['obs'])
    self._act, _ = torch.nn.utils.rnn.pad_packed_sequence(rollout_data['act'])

    self._N = len(self._ep_lens)

    # Shuffle the data.
    shuffle_indices = np.random.choice(range(self.N), size=self.N, replace=False)
    self._obs = self._obs[:, shuffle_indices]
    self._act = self._act[:, shuffle_indices]
    self._ep_lens = self._ep_lens[shuffle_indices]

  def add_data(self, rollout_data):
    # TODO(eholly): Is 0 the right dim throughout this method? Is it time-major or batch-major?
    assert isinstance(rollout_data['obs'], PackedSequence)
    assert isinstance(rollout_data['act'], PackedSequence)

    obs, new_ep_len = pad_packed_sequence(rollout_data['obs'])
    act, _ = pad_packed_sequence(rollout_data['act'])
    max_new_ep_len = torch.max(new_ep_len)

    # If existing data time dim is not long enough, pad it to required length.
    if max_new_ep_len > self._obs.shape[0]:
      obs_extra_padding_shape = list(self._obs.shape)
      obs_extra_padding_shape[0] = max_new_ep_len - self._obs.shape[0]
      self._obs = torch.cat([self._obs, torch.zeros(obs_extra_padding_shape)], dim=0)
      act_extra_padding_shape = list(self._act.shape)
      act_extra_padding_shape[0] = max_new_ep_len - self._act.shape[0]
      self._act = torch.cat([self._act, torch.zeros(act_extra_padding_shape)], dim=0)

    # If new data time dim is not long enough, pad it to required length.
    if max_new_ep_len < self._obs.shape[0]:
      obs_extra_padding_shape = list(obs.shape)
      obs_extra_padding_shape[0] = self._obs.shape[0] - max_new_ep_len
      obs = torch.cat([obs, torch.zeros(obs_extra_padding_shape)], dim=0)
      act_extra_padding_shape = list(act.shape)
      act_extra_padding_shape[0] = self._act.shape[0] - max_new_ep_len
      act = torch.cat([act, torch.zeros(act_extra_padding_shape)], dim=0)

    self._obs = torch.cat([obs, self._obs], dim=1)  # Cat on batch dim.
    self._act = torch.cat([act, self._act], dim=1)  # Cat on batch dim.
    self._ep_lens = torch.cat([new_ep_len, self._ep_lens])
    self._N += obs.shape[1]

    if self._N > self._max_size:
      self._N = self._max_size
      self._obs = self._obs[:self._N]
      self._act = self._act[:self._N]
      self._ep_lens = self._ep_lens[:self._N]

      self._update_train_cutoff()

  def sample(self, batch_size=None, eval=False):
    if batch_size is None:
      batch_size = self._batch_size
    
    # Choose sample indices.
    if not eval:
      sample_range = range(self._train_cutoff)
    else:
      sample_range = range(self._train_cutoff, self.N)
    sample_size = min(batch_size, len(sample_range))
    batch_indices = np.random.choice(sample_range, size=sample_size, replace=False)

    # Take data from replay buffer.
    sample_ep_lens = self._ep_lens[batch_indices]
    sample_obs, sample_act = self._obs[:, batch_indices], self._act[:, batch_indices]

    # Sort into shortest episodes.
    sample_ep_lens, sort_idxs = torch.sort(sample_ep_lens, descending=True)
    sample_obs, sample_act = sample_obs[:, sort_idxs], sample_act[:, sort_idxs]

    # Switch from padded to PackedSequence.
    sample_data = {
      'obs': torch.nn.utils.rnn.pack_padded_sequence(sample_obs, sample_ep_lens),
      'act': torch.nn.utils.rnn.pack_padded_sequence(sample_act, sample_ep_lens),
    }

    return sample_data

DEFAULT_BC_LOSS_FN = lambda x, y: torch.mean(torch.pow(x-y, 2.0))
class BCTrainer(trainer.Trainer):

  def __init__(self, *args, **kwargs):
    if 'loss_fn' not in kwargs or kwargs['loss_fn'] is None:
      self._loss_fn = DEFAULT_BC_LOSS_FN
    if 'loss_fn' in kwargs:
      if kwargs['loss_fn'] is not None:
        self._loss_fn = kwargs['loss_fn']
      del kwargs['loss_fn']
    super().__init__(*args, **kwargs)
    assert issubclass(type(self._model), policy.TorchPolicy)

  @property
  def policy(self):
    return self.model

  def _parameters(self):
    return [self._model.parameters()]

  def _inference_and_loss(self, sample_data):
    pred_act = self.policy(sample_data['obs'].data)
    expert_act = sample_data['act'].data
    loss = self._loss_fn(pred_act, expert_act)
    summaries.add_scalar('_performance/loss', loss, self.global_step)
    return loss,

class ReflexBCTrainer(trainer.Trainer):

  def __init__(self, *args, **kwargs):
    if 'loss_fn' not in kwargs or kwargs['loss_fn'] is None:
      self._loss_fn = DEFAULT_BC_LOSS_FN
    if 'loss_fn' in kwargs:
      if kwargs['loss_fn'] is not None:
        self._loss_fn = kwargs['loss_fn']
      del kwargs['loss_fn']
    super().__init__(*args, **kwargs)
    assert issubclass(type(self._model), policy.ReflexPolicy)

  @property
  def policy(self):
    return self.model

  def _parameters(self):
    return list(self._model.parameters())

  def _inference_and_loss(self, sample_data):
    # Compute bc_loss.
    pred_act = self.policy(sample_data['obs'])
    expert_act = sample_data['act']
    bc_loss = self._loss_fn(pred_act, expert_act)
    summaries.add_scalar('_performance/bc_loss', bc_loss, self.global_step)

    # Compute reflex_loss.
    reflex_outputs = self.policy.reflex_outputs(sample_data['obs'])
    unweighted_reflexes_loss = self._loss_fn(reflex_outputs, torch.unsqueeze(expert_act, dim=-1))
    reflex_softmax_weights = self.policy.reflex_softmax_weights(sample_data['obs'])
    weighted_reflexes_loss = unweighted_reflexes_loss * reflex_softmax_weights
    reflexes_loss = torch.mean(torch.sum(weighted_reflexes_loss, dim=-1))

    summaries.add_scalar('_performance/reflexes_loss', reflexes_loss, self.global_step)
    summaries.add_histogram('reflexes/softmax_weights', reflex_softmax_weights, self.global_step)

    reflex_conditional_entropy = -torch.sum(reflex_softmax_weights * torch.log(reflex_softmax_weights), dim=-1)
    summaries.add_histogram('reflexes/reflex_conditional_entropy', reflex_conditional_entropy, self.global_step)
    reflex_marginals = torch.mean(reflex_softmax_weights, dim=0)
    reflex_marginal_entropy = -torch.sum(reflex_marginals * torch.log(reflex_marginals), dim=-1)
    summaries.add_histogram('reflexes/reflex_marginal_entropy', reflex_marginal_entropy, self.global_step)

    supervisor_loss = bc_loss

    return supervisor_loss, reflexes_loss

class SoftKNNBCTrainer(trainer.Trainer):

  def __init__(self, *args, **kwargs):
    if 'loss_fn' not in kwargs or kwargs['loss_fn'] is None:
      self._loss_fn = DEFAULT_BC_LOSS_FN
    if 'loss_fn' in kwargs:
      if kwargs['loss_fn'] is not None:
        self._loss_fn = kwargs['loss_fn']
      del kwargs['loss_fn']
    super().__init__(*args, **kwargs)
    assert issubclass(type(self._model), policy.SoftKNNPolicy), (
      "Got policy of type %s, expected SoftKNNPolicy." % type(self._model)
    )
    
  def _initialize(self, dataset):
    training_data = dataset.sample(batch_size=float('inf'))
    obs_data = training_data['obs']
    obs_mean = obs_data.mean(dim=0)
    obs_stddev = obs_data.std(dim=0)
    act_data = training_data['act']
    act_mean = act_data.mean(dim=0)
    act_stddev = act_data.std(dim=0)
    self.policy.model.initialize_points(
      input_mean=obs_mean,
      input_stddev=obs_stddev,
      output_mean=act_mean,
      output_stddev=act_stddev,
    )

  def _train(self, sample_data=None):
    if sample_data is None:
      sample_data = self._dataset.sample()
    
    losses = super()._train(sample_data=sample_data)
    
    mse_loss = losses[0]
    _, top_loss_idx = torch.topk(mse_loss, 1)
    self.policy.model.set_point(
      sample_data['obs'][top_loss_idx], sample_data['act'][top_loss_idx])

    return losses

  @property
  def policy(self):
    return self.model

  def _parameters(self):
    return [self._model.parameters()]

  def _inference_and_loss(self, sample_data):
    self.policy.model.global_step = self.global_step

    pred_act = self.policy(sample_data['obs'])
    expert_act = sample_data['act']

    # loss = self._loss_fn(pred_act, expert_act)

    # Sum over action dim, but not batch dim.
    loss = torch.sum(torch.pow(pred_act - expert_act, 2.0), dim=-1)

    return loss,
