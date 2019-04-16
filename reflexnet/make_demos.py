
import argparse
import gym
import os
import roboschool
import torch

import rollouts
import utils

def make_demos(
  log_dir,
  env_name,
  action_noise,
  max_rollout_length,
  num_rollouts,
  demo_filename,
):
  # Make environment.
  env = gym.make(env_name)

  # Make policy.
  policy = utils.make_roboschool_policy(env_name, env)

  # Make rollout data.
  rollout_data = rollouts.rollout_n(
    num_rollouts,
    env=env,
    policy=policy,
    max_steps=max_rollout_length,
    action_noise=action_noise,
    )

  # Save demos to file.
  save_path = os.path.join(log_dir, demo_filename)
  torch.save(rollout_data, save_path)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', required=True, type=str, help='Parent directory under which to save output.')
  parser.add_argument('--env_name', default='RoboschoolWalker2d-v1', type=str, help='Name of env to make and use.')
  parser.add_argument('--action_noise', default=0.3, type=float, help='Probability of adding noise to each action.')
  parser.add_argument('--max_rollout_length', default=200, type=int, help='Max length of each rollout.')
  parser.add_argument('--num_rollouts', default=6000, type=int, help='Number of rollouts to collect and save.')
  parser.add_argument('--demo_filename', default='rollouts.torch', type=str, help='Name of demo file to save.')
  args = parser.parse_args()

  args.log_dir = os.path.join(args.log_dir, 'demos', args.env_name)

  # Make a unique directory under log_dir and save args there.
  utils.init_log_dir(args)

  make_demos(**vars(args))

if __name__ == '__main__':
  main()