from abc import ABC, abstractmethod
import os
import time
from tqdm import tqdm

import summaries
import tensorflow as tf
import utils

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

  def __init__(self, model, dataset, optim_cls=tf.train.AdamOptimizer, learning_rate=1e-5):
    self._model = model
    assert issubclass(dataset.__class__, Dataset)
    self._dataset = dataset
    self._optimizers = [
      optim_cls(learning_rate=learning_rate)
      for _ in self._parameters()
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
    with tf.compat.v1.Session().as_default():
      if eval_every is None:
        eval_every = int(train_steps / 20)
      self._global_step = 0

      # Make data input placeholders.
      sample_data = self._dataset.sample()
      eval_sample_data = self._dataset.sample(batch_size=float('inf'), eval=True)
      self._sample_placeholder = utils.tf_placeholder_for_tensor(sample_data)
      self._eval_sample_placeholder = utils.tf_placeholder_for_tensor(eval_sample_data)

      # Create loss tensors.
      self._train_losses = self._inference_and_loss(self._sample_placeholder)
      self._eval_losses = self._inference_and_loss(self._eval_sample_placeholder)

      # Create train_op.
      minimize_ops = [
        opt.minimize(loss, var_list=params)
        for opt, params, loss in zip(self._optimizers, self._parameters(), self._train_losses)
        ]
      self._train_op = tf.group(minimize_ops)

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

  def _train(self):
    start_time = time.time()
    sample_data = self._dataset.sample()

    train_outputs = tf.compat.v1.get_default_session().run(
      [self._train_op] + self._train_losses,
      feed_dict={self._sample_placeholder: sample_data}
    )
    loss_values = train_outputs[1:]

    # Summarize timing.
    total_time = time.time() - start_time
    steps_per_sec = 1 / total_time
    summaries.add_scalar('misc/train_steps_per_sec', steps_per_sec, self.global_step)

    return loss

  def _eval(self):
    with torch.no_grad():
      start_time = time.time()
      sample_data = self._dataset.sample(batch_size=float('inf'), eval=True)
      loss_values = tf.compat.v1.get_default_session().run(
        self._eval_losses,
        feed_dict={self._eval_sample_placeholder: sample_data}
      )
      total_time = time.time() - start_time

    # Summarize timing.
    steps_per_sec = 1 / total_time
    summaries.add_scalar('misc/eval_steps_per_sec', steps_per_sec, self.global_step)
    
    return loss