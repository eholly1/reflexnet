
import numpy as np
import torch
import utils

def rollout(env, policy, max_steps=1000, action_noise=0.0):
  """Run one rollout and return data.

  Args:
    env: The environment with reset and step function.
    policy: The state->action policy to roll out.
    max_steps: The maximum number of steps in the episode.
    action_noise: The probability of adding noise to the action
      before sending to the environment. Action noise does not
      get logged in rollout data.
  Returns:
    A dict of torch tensors, with time dimension for 'obs', 'act'
      'rew' and 'done'.
  """
  rollout_data = {
    'obs': [],
    'act': [],
    'rew': [],
    'done': [],
    'total_rew': 0.0,
    'num_steps': 0,
  }

  # Initialize collection.
  obs = env.reset()
  rollout_data['obs'].append(obs)
  done = False

  while not done and rollout_data['num_steps'] < max_steps:
    rollout_data['num_steps'] += 1
    act = policy(obs)
    rollout_data['act'].append(act)
    if action_noise > 0.0:
      if np.random.uniform() < action_noise:
        act += env.action_space.sample()
    obs, rew, done, _ = env.step(act)

  rollout_data = utils.tree_apply(torch.tensor, rollout_data)
  for k in ['obs', 'act', 'rew', 'done']:
    rollout_data[k] = torch.stack(rollout_data[k])
  return rollout_data

def rollout_n(n, *args, **kwargs):
  """Run multiple rollouts and batch the results.
  Args:
    n: The number of rollouts.
  """
  rollout_data_list = []
  print('Collectin %d episodes...'%n)
  for _ in tdqm(range(n)):
    rollout_data_list.append(rollout_episode(*args, **kwargs))

  # Batch the data.
  batched_rollout_data = {}
  for k in rollout_data_list[0]:
    batched_rollout_data[k] = torch.stack([rollout_data[k] for rollout_data in rollout_data_list])

  # Make the data time-major.
  for k in ['obs', 'act', 'rew', 'done']:
    batched_rollout_data[k] = torch.transpose(batched_rollout_data[k], 0, 1)

  return batched_rollout_data