import numpy as np
import os
import torch

import summaries
import trainer
import utils

# TODO(eholly1): Make generic torch in-memory Dataset class.
class BCFrameDataset(trainer.Dataset):

  def __init__(self, batch_size, load_path, eval_fraction=0.2):
    self._batch_size = batch_size

    if os.path.isfile(load_path)
      rollout_data = torch.load(load_path)
    elif os.path.isdir(load_path):
      rollout_data_list = utils.load_subdirs(load_path)
    else:
      raise ValueError('Load path not found: %s' % load_path)

    self._init_dataset_with_rollout_data(rollout_data)
    self._train_cutoff = int(self.N * (1.0 - eval_fraction))
    
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

  @property
  def policy(self):
    return self.model

  def _inference_and_loss(self, sample_data):
    pred_act = self.policy(sample_data['obs'])
    expert_act = sample_data['act']
    loss = self._loss_fn(pred_act, expert_act)
    summaries.add_scalar('_performance/loss', loss, self.global_step)
    return loss

