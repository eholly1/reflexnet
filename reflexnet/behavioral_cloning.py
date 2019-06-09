import numpy as np
import os
import torch

import policy
import summaries
import trainer
import utils

# TODO(eholly1): Make generic torch in-memory Dataset class.
class BCFrameDataset(trainer.Dataset):

  def __init__(self, batch_size, load_path, eval_fraction=0.2):
    self._batch_size = batch_size

    if os.path.isfile(load_path):
      rollout_data = torch.load(load_path)
    elif os.path.isdir(load_path):
      rollout_data_list = utils.load_subdirs(load_path)
      rollout_data = utils.tree_apply(utils.merge_packed_sequences, *rollout_data_list)
    else:
      raise ValueError('Load path not found: %s' % load_path)
    self._init_dataset_with_rollout_data(rollout_data)
    self._eval_fraction = eval_fraction
    self._train_cutoff = int(self.N * (1.0 - self._eval_fraction))
    
  @property
  def N(self):
    return self._N

  def _init_dataset_with_rollout_data(self, rollout_data):
    assert isinstance(rollout_data['obs'], torch.nn.utils.rnn.PackedSequence)
    assert isinstance(rollout_data['act'], torch.nn.utils.rnn.PackedSequence)

    # Store states and actions with single batch dim.
    self._obs = rollout_data['obs'].data
    self._act = rollout_data['act'].data
    assert self._act.shape[0] == self._obs.shape[0]
    assert len(self._act.shape) == 2
    self._N = self._obs.shape[0]

    # Shuffle the data.
    shuffle_indices = np.random.choice(range(self.N), size=self.N, replace=False)
    self._obs = self._obs[shuffle_indices, :]
    self._act = self._act[shuffle_indices, :]

  def add_data(self, rollout_data):
    assert isinstance(rollout_data['obs'], torch.Tensor)
    assert isinstance(rollout_data['act'], torch.Tensor)
    self._obs = torch.cat([rollout_data['obs'], self._obs], dim=0)
    self._act = torch.cat([rollout_data['act'], self._act], dim=0)
    self._N += rollout_data['obs'].shape[0]
    self._train_cutoff = int(self.N * (1.0 - self._eval_fraction))

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

    sample_data = {
      'obs': self._obs[batch_indices],
      'act': self._act[batch_indices],
    }
    return sample_data

# DEFAULT_BC_LOSS_FN = torch.nn.SmoothL1Loss()
DEFAULT_BC_LOSS_FN = torch.nn.MSELoss()
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
    pred_act = self.policy(sample_data['obs'])
    expert_act = sample_data['act']
    loss = self._loss_fn(pred_act, expert_act)
    summaries.add_scalar('_performance/loss', loss, self.global_step)
    return loss

# DEFAULT_BC_LOSS_FN = torch.nn.SmoothL1Loss()
DEFAULT_BC_LOSS_FN = torch.nn.MSELoss()
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

    reflex_outputs = self.policy.reflex_outputs(sample_data['obs'])
    unweighted_reflexes_loss = self._loss_fn(reflex_outputs, torch.unsqueeze(expert_act, dim=-1))
    reflex_softmax_weights = self.policy.reflex_softmax_weights(sample_data['obs'])
    weighted_reflexes_loss = unweighted_reflexes_loss * reflex_softmax_weights
    reflexes_loss = torch.mean(torch.sum(weighted_reflexes_loss, dim=-1))

    summaries.add_scalar('_performance/reflexes_loss', reflexes_loss, self.global_step)
    summaries.add_histogram('reflexes/softmax_weights', reflex_softmax_weights, self.global_step)

    return bc_loss, reflexes_loss