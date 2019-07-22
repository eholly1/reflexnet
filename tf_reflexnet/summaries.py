"""This module is for summary & scope management using contexts.

Users may add summaries under a scope for summaries to be written under that scope path.

If more than one scalar summary is written with the same name under a single scope, the
values will be averaged, and the global_step will be the maximum of the global steps written.

Each scope can be constructed with a name and a path. The path is appended to the log_dir for
where summaries will be written, e.g. 'train' or 'eval'. This will create separate cuves for
separate paths. The name is appended to the name of the summary, which is also used for
tabs and categorization within tensorboard.

Example:

with summaries.Scope('train'):
  with summaries.Scope('losses'):
    summaries.add_scalar('loss1', 3.0, 1)
    summaries.add_scalar('loss2', 5.0, 1)
  # When a scope exits, all summaries from within that scope will be written
  #   under the path. In this example summaries will be:
  #   - 'train/losses/loss1': 3.0 at global_step 1
  #   - 'train/losses/loss2': 5.0 at global_step 1

  summaries.add_scalar('other_scalar', 7.0, 5)
# Now the summary 'train/other_scalar': 7.0 at global_step 5 will be written.

"""

import os
import torch
from tensorboardX import SummaryWriter

_summary_log_dir = None
def init_summary_log_dir(log_dir):
  global _summary_log_dir
  if _summary_log_dir is not None:
    raise ValueError('Summary log dir already initialized to "%s".' % _summary_log_dir)
  print('Saving summaries to: {}'.format(log_dir))
  _summary_log_dir = log_dir

def _add_summary_value(add_fn):
  global _active_scopes
  if len(_active_scopes) == 0:
    with Scope() as current_scope:
      add_fn(current_scope)
  else:
    current_scope = _active_scopes[-1]
    add_fn(current_scope)

def add_scalar(name, value, global_step):
  _add_summary_value(lambda scope: scope._add_scalar(name, value, global_step))

def add_histogram(name, value, global_step):
  _add_summary_value(lambda scope: scope._add_histogram(name, value, global_step))


_active_scopes = []
class Scope(object):
  def __init__(self, name='', path=''):
    global _summary_log_dir
    if _summary_log_dir is None:
      raise ValueError('Please call summaries.init_summary_log_dir before constructing scope.')

    self._name = name
    self._path = path
    self._scalar_summaries = {}
    self._histogram_summaries = {}

  def _add_scalar(self, name, value, global_step):
    if name not in self._scalar_summaries:
      self._scalar_summaries[name] = (value, global_step, 1)
    else:
      old_value, old_global_step, old_count = self._scalar_summaries  [name]
      new_count = old_count + 1
      new_value = (old_value * old_count + value) / new_count
      new_global_step = max(old_global_step, global_step)
      self._scalar_summaries[name] = (new_value, new_global_step, new_count)

  def _add_histogram(self, name, value, global_step):
    if name not in self._histogram_summaries:
      self._histogram_summaries[name] = (
        torch.unsqueeze(value, dim=-1), global_step)
    else:
      old_value, old_global_step = self._histogram_summaries[name]
      self._histogram_summaries[name] = (
        torch.cat([old_value, torch.unsqueeze(value, dim=-1)], dim=-1),
        max(old_global_step, global_step)
        )

  def __enter__(self):
    global _active_scopes
    if _active_scopes:
      self._name = os.path.join(_active_scopes[-1]._name, self._name)
      self._path = os.path.join(_active_scopes[-1]._path, self._path)
    else:
      # First scope on the stack gets path prepended with _summary_log_dir.
      self._path = os.path.join(_summary_log_dir, self._path)
    _active_scopes.append(self)
    return self

  def __exit__(self, error_type, error_value, error_traceback):
    global _active_scopes
    assert _active_scopes[-1] == self
    summary_writer = SummaryWriter(self._path)
    for name, (value, global_step, _) in self._scalar_summaries.items():
      summary_writer.add_scalar(os.path.join(self._name, name), value, global_step)
    for name, (value, global_step) in self._histogram_summaries.items():
      summary_writer.add_histogram(os.path.join(self._name, name), value, global_step)
    del _active_scopes[-1]  
  