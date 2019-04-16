
import numpy as np
import torch
from tqdm import tqdm
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
    rollout_data['obs'].append(obs)
    rollout_data['rew'].append(rew)
    rollout_data['done'].append(done)

  rollout_data = utils.tree_apply(torch.tensor, rollout_data)
  for k in ['obs', 'act', 'rew', 'done']:
    rollout_data[k] = torch.stack(rollout_data[k])
  return rollout_data

def rollout_n(n, *args, **kwargs):
  """Run multiple rollouts and batch the results.
  Args:
    n: The number of rollouts.
  Returns:
    A dict of packed sequences for keys ['obs', 'act', 'rew', 'done'].
  """
  rollout_data_list = []
  print('Collectin %d episodes...'%n)
  for _ in tqdm(range(n)):
    rollout_data_list.append(rollout(*args, **kwargs))

  # Batch the data.
  sort_indices = torch.argsort(
    torch.tensor([rollout['num_steps'] for rollout in rollout_data_list]),
    descending=True)
  rollout_data_list = [rollout_data_list[i] for i in sort_indices]
  packed_rollout_data = {}
  for k in ['obs', 'act', 'rew', 'done']:
    packed_rollout_data[k] = torch.nn.utils.rnn.pack_sequence(
      [rollout_data[k] for rollout_data in rollout_data_list])

  return packed_rollout_data