import numpy as np
import os
import torch

import summaries
import trainer
import utils


EMBEDDING_DISTANCE = 1.0
DISTANCE_METRIC = torch.nn.MSELoss()
class PolicyEmbeddingTrainer(trainer.Trainer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  def policy(self):
    return self.model

  def _inference_and_loss(self, sample_data):
    embeddings = self.policy.get_embedding(sample_data['obs'])
    

    pred_act = self.policy(sample_data['obs'])
    expert_act = sample_data['act']
    loss = self._loss_fn(pred_act, expert_act)
    summaries.add_scalar('_performance/loss', loss, self.global_step)
    return loss

