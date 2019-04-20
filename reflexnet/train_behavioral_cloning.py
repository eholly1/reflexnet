import argparse
import gym
import os
import roboschool
import torch

from behavioral_cloning import BCTrainer, BCFrameDataset
import policy
import rollouts
import summaries
import utils

def train_behavioral_cloning(
  log_dir,
  env_name,
  demo_filepath,
  batch_size,
  learning_rate,
  train_steps,
  eval_every,
  ):
  summaries.init_summary_log_dir(log_dir)
  dataset = BCFrameDataset(batch_size, demo_filepath)
  env = gym.make(env_name)
  training_policy = policy.FeedForwardPolicy.for_env(env)
  trainer = BCTrainer(
    model=training_policy,
    dataset=dataset,
    learning_rate=learning_rate,
    )

  def _after_eval_callback():
    n = 20
    packed_rollout_data = rollouts.rollout_n(n, env, training_policy)
    avg_rew = torch.sum(packed_rollout_data['rew'].data) / n
    trainer.print('Avg rollout reward: ', avg_rew)
    summaries.add_scalar('_performance/avg_rollout_reward', avg_rew, trainer.global_step)

  trainer.train_and_eval(
    log_dir=log_dir,
    train_steps=train_steps,
    after_eval_callback=_after_eval_callback,
  )

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', required=True, type=str, help='Parent directory under which to save output.')
  parser.add_argument('--env_name', default='RoboschoolWalker2d-v1', type=str, help='Parent directory under which to save output.')
  parser.add_argument('--demo_filepath', required=True, type=str, help='Full path to file with task demos.')
  parser.add_argument('--batch_size', default=16, type=int, help='Batch size for SGD.')
  parser.add_argument('--learning_rate', default=1e-7, type=int, help='Learning rate for optimizer.')
  parser.add_argument('--train_steps', default=30000, type=int, help='Total number of train steps.')
  parser.add_argument('--eval_every', default=None, type=int, help='Eval after this many train steps.')
  args = parser.parse_args()

  args.log_dir = os.path.join(args.log_dir, 'behavioral_cloning', args.env_name)

  utils.init_log_dir(args)

  train_behavioral_cloning(**vars(args))

if __name__ == "__main__":
  main()