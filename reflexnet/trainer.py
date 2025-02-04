from abc import ABC, abstractmethod
import os
import time
from tqdm import tqdm

import summaries
import torch

class Dataset(ABC):

  @property
  @abstractmethod
  def N(self):
    raise NotImplementedError

  @abstractmethod
  def sample(self, batch_size=None, eval=False):
    raise NotImplementedError

class Trainer(ABC):
  """This class orchestrates training, including managing:
    - Train / test frequenceis.
    - Calling gradients / optimizer.
    - Checkpointing.
    - Gathering and saving summaries.
  """
  # TODO(eholly1): Checkpointing.
  # TODO(eholly1): Summaries.

  def __init__(self, model, dataset, optim_cls=torch.optim.Adam, learning_rate=1e-5):
    assert issubclass(type(model), torch.nn.Module)
    self._model = model
    assert issubclass(dataset.__class__, Dataset)
    self._dataset = dataset
    self._optimizers = [
      optim_cls(params=p, lr=learning_rate)
      for p in self._parameters()
    ]
    self._global_step = 0

  @abstractmethod
  def _parameters(self):
    """Get the parameters to train.
      Returns: A list of parameter iterators. Each iterator in
        the list should be given its own optimizer. The the parameter
        iterators in order correspond to the loss functions returned by
        inference_and_loss.
    """
    raise NotImplementedError

  @abstractmethod
  def _inference_and_loss(self, sample_data):
    """Perform inference and compute loss on samples.
    Args:
      sample_data: A sample of data with which to compute loss.
    Returns:
      A list of loss Tensors, corresponding element-wise to the list of
        optimizers.
    """
    raise NotImplementedError

  def _initialize(self, dataset):
    """Perform any training initialization needed."""
    pass

  @property
  def model(self):
    return self._model

  @property
  def global_step(self):
    return self._global_step

  def train_and_eval(
    self,
    log_dir,
    train_steps,
    eval_every=None,
    after_eval_callback=None,
    ):
    if eval_every is None:
      eval_every = int(train_steps / 20)
    self._global_step = 0

    self._initialize(self._dataset)

    # Initial eval.
    self.print('Running initial eval.')
    with summaries.Scope(path='eval'):
      self._eval()
    if after_eval_callback:
        after_eval_callback()

    self._model.save(os.path.join(log_dir, 'policy'))    

    # Training Iterations
    while self.global_step < train_steps:

      # Run training.
      self.print('Running training.')
      with summaries.Scope(path='train'):
        for _ in tqdm(range(eval_every)):
          self._global_step += 1
          self._train()
          if self.global_step >= train_steps:
            break

      # Perform eval.
      self.print('Running eval.')
      with summaries.Scope(path='eval'):
        self._eval()

      self._model.save(os.path.join(log_dir, 'policy'))

      # After eval callback.
      if after_eval_callback:
        after_eval_callback()

  def print(self, *args):
    args = ["[{}]\t".format(self.global_step)] + list(args)
    print_str = ("{}" * len(args)).format(*args)
    print(print_str)

  def _train(self, sample_data=None):
    start_time = time.time()
    self._model.train()  # Put model in train mode.
    if sample_data is None:
      sample_data = self._dataset.sample()
    losses = self._inference_and_loss(sample_data)
    for i, (opt, loss) in enumerate(zip(self._optimizers, losses)):
      loss = torch.mean(loss)
      summaries.add_scalar('_performance/loss', loss, self.global_step)

      opt.zero_grad()
      loss.backward(retain_graph=(i+1 != len(losses)))
      opt.step()

    # Summarize timing.
    total_time = time.time() - start_time
    steps_per_sec = 1 / total_time
    summaries.add_scalar('misc/train_steps_per_sec', steps_per_sec, self.global_step)

    if hasattr(self._model, "reset"):
      self._model.reset()

    return losses

  def _eval(self):
    with torch.no_grad():
      self._model.eval()  # Put model in eval mode.
      start_time = time.time()
      sample_data = self._dataset.sample(batch_size=float('inf'), eval=True)
      losses = self._inference_and_loss(sample_data)
      total_time = time.time() - start_time

    # Summarize timing.
    steps_per_sec = 1 / total_time
    summaries.add_scalar('misc/eval_steps_per_sec', steps_per_sec, self.global_step)

    if hasattr(self._model, "reset"):
      self._model.reset()
    
    return losses