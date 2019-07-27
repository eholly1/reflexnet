import argparse
import gym
import os
import roboschool
import torch

import behavioral_cloning
import policy
import rollouts
import summaries
import utils

TRAINING_CLASSES = {
  'MLP': (behavioral_cloning.BCTrainer, policy.FeedForwardPolicy),
  'Reflex': (behavioral_cloning.ReflexBCTrainer, policy.ReflexPolicy),
  'SoftKNN': (behavioral_cloning.SoftKNNBCTrainer, policy.SoftKNNPolicy),
}

def train_daggr(
  log_dir,
  env_name,
  demo_filepath,
  batch_size,
  learning_rate,
  train_steps,
  eval_every,
  training_type,
  ):
  trainer_cls, policy_cls = TRAINING_CLASSES[training_type]
  summaries.init_summary_log_dir(log_dir)
  dataset = behavioral_cloning.BCFrameDataset(batch_size, demo_filepath)
  env = gym.make(env_name)
  training_policy = policy_cls.for_env(env)

  import numpy as np
  total_params = 0
  for pg in training_policy.parameters():
    for p in pg:
      total_params += np.product(p.shape) 
  trainer = trainer_cls(
    model=training_policy,
    dataset=dataset,
    learning_rate=learning_rate,
    )

  oracle_policy = utils.make_roboschool_policy(env_name, env)

  def _after_eval_callback():
    n = 100
    packed_rollout_data = rollouts.rollout_n(n, env, training_policy)
    avg_rew = torch.sum(packed_rollout_data['rew'].data) / n
    trainer.print('Avg rollout reward: ', avg_rew)
    summaries.add_scalar('_performance/avg_rollout_reward', avg_rew, trainer.global_step)

    # Add oracle data to dataset.
    oracle_actions = oracle_policy(packed_rollout_data['obs'].data)
    oracle_data = {
      'obs': packed_rollout_data['obs'].data,
      'act': torch.tensor(oracle_actions),
    }
    dataset.add_data(oracle_data)

  trainer.train_and_eval(
    log_dir=log_dir,
    train_steps=train_steps,
    after_eval_callback=_after_eval_callback,
    eval_every=eval_every,
  )

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', required=True, type=str, help='Parent directory under which to save output.')
  parser.add_argument('--env_name', default='RoboschoolWalker2d-v1', type=str, help='Parent directory under which to save output.')
  parser.add_argument('--demo_filepath', required=True, type=str, help='Full path to file with task demos.')
  parser.add_argument('--training_type', default='MLP', type=str, help='The type of policy to train.')
  parser.add_argument('--batch_size', default=64, type=int, help='Batch size for SGD.')
  parser.add_argument('--learning_rate', default=1e-5, type=int, help='Learning rate for optimizer.')
  parser.add_argument('--train_steps', default=30000, type=int, help='Total number of train steps.')
  parser.add_argument('--eval_every', default=None, type=int, help='Eval after this many train steps.')
  args = parser.parse_args()

  args.log_dir = os.path.join(args.log_dir, 'daggr', args.env_name)

  utils.init_log_dir(args)

  train_daggr(**vars(args))

if __name__ == "__main__":
  main()