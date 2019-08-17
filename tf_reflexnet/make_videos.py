
import argparse
import gym
import os
import numpy as np
import roboschool
import torch

import policy as torch_policy
import rollouts
import utils

def make_videos(
  render_dir,
  policy_path,
  env_name,
  action_noise,
  max_rollout_length,
  num_rollouts,
):
  # Make environment.
  env = gym.make(env_name)

  # Make policy.
  if policy_path is None:
    policy = utils.make_roboschool_policy(env_name, env)
  else:
    policy = torch_policy.TorchPolicy.load(policy_path)

  # Make rollout data.
  rollouts.rollout_n(
    num_rollouts,
    env=env,
    policy=policy,
    max_steps=max_rollout_length,
    action_noise=action_noise,
    render_dir=render_dir,
    )

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--render_dir', required=True, type=str, help='Directory in which to save output.')
  parser.add_argument('--policy_path', default=None, type=str, help='Path to policy file or dir containing policy files.')
  parser.add_argument('--env_name', default='RoboschoolWalker2d-v1', type=str, help='Name of env to make and use.')
  parser.add_argument('--action_noise', default=0.0, type=float, help='Probability of adding noise to each action.')
  parser.add_argument('--max_rollout_length', default=1000, type=int, help='Max length of each rollout.')
  parser.add_argument('--num_rollouts', default=10, type=int, help='Number of rollouts to collect and save.')
  args = parser.parse_args()

  make_videos(**vars(args))

if __name__ == '__main__':
  main()