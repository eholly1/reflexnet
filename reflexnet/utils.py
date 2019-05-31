
import json
import os
import subprocess
import tensorflow as tf
import torch
from torch.nn.utils.rnn import PackedSequence

import RoboschoolWalker2d_v1_2017jul

def merge_packed_sequences(*packed_sequences):
  max_length = max(*[len(s.batch_sizes) for s in packed_sequences])
  padded_sequences_and_lengths = [
    torch.nn.utils.rnn.pad_packed_sequence(s, total_length=max_length)
    for s in packed_sequences]
  padded_sequences, lengths = zip(*padded_sequences_and_lengths)
  padded_sequences = torch.cat(padded_sequences, dim=1)  # Cat padded_sequences on batch dim.
  lengths = torch.cat(lengths)
  sorted_lengths, sort_indices = torch.sort(lengths, descending=True)
  sorted_padded_sequences = padded_sequences[:, sort_indices]
  return torch.nn.utils.rnn.pack_padded_sequence(sorted_padded_sequences, sorted_lengths)


def load_subdirs(load_path):
  if os.path.isfile(load_path):
    if load_path.endswith('.torch'):
      return [torch.load(load_path)]
    else:
      return []
  elif os.path.isdir(load_path):
    ret_list = []
    for fname in os.listdir(load_path):
      ret_list.extend(load_subdirs(os.path.join(load_path, fname)))
    return ret_list
  else:
    raise ValueError('Load path not found: %s' % load_path)

def make_roboschool_policy(env_name, env):
  if env_name == "RoboschoolWalker2d-v1":
    policy_cls = RoboschoolWalker2d_v1_2017jul.ZooPolicyTensorflow
  else:
    raise ValueError("Unsupported env_name", env_name)

  config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count = { "GPU": 0 } )
  tf.InteractiveSession(config=config)
  zoo_policy = policy_cls("mymodel1", env.observation_space, env.action_space)
  zoo_policy.old_act = zoo_policy.act
  zoo_policy.act = lambda obs: zoo_policy.old_act(obs, env)

  return zoo_policy

def tree_apply(fn, *tree_nodes):
  if isinstance(tree_nodes[0], dict):
    ret_val = {}
    for k in tree_nodes[0]:
      ret_val[k] = tree_apply(fn, *[tn[k] for tn in tree_nodes])
    return ret_val
  if isinstance(tree_nodes[0], list):
    ret_val = []
    for i in range(len(tree_nodes[0])):
      ret_val.append(tree_apply(fn, *[tn[i] for tn in tree_nodes]))
    return ret_val
  if isinstance(tree_nodes[0], torch.Tensor) or isinstance(tree_nodes[0], PackedSequence):
    return fn(*tree_nodes)
  return fn(*tree_nodes)

def _confirm(question):
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("%s [y/n]: " % question).lower()
    return answer == "y"

def init_log_dir(args):
  # Make sure changes are committed, or warn user.
  git_status_output = str(subprocess.check_output(["git", "status"]))
  without_git = False
  if "nothing to commit, working tree clean" not in git_status_output:
    if not _confirm("Not all changes committed. Run training without logging git hash?"):
      print("Exiting...")
      os._exit(0)
    without_git = True
  
  # Create logdir with unique postfix id.
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
  if not os.path.isdir(args.log_dir):
    raise ValueError("log_dir filename taken '%s'" % args.log_dir)
  subdir_i = 0
  while os.path.exists(os.path.join(args.log_dir, str(subdir_i))):
    subdir_i += 1
  args.log_dir = os.path.join(args.log_dir, str(subdir_i))
  os.makedirs(args.log_dir)

  # Save Git commit hash.
  if not without_git:
    git_log_output = str(subprocess.check_output(["git", "log", "--pretty=oneline"]))
    commit_hash = git_log_output.split(" ")[0]
    commit_message = "_".join(git_log_output.split(".")[0].split(" ")[1:])[:75]
    args.log_dir = os.path.join(args.log_dir, commit_message)
    os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'git_info.json'), "w") as git_info_file:
      git_info = {
        'commit_hash': commit_hash,
      }
      json.dump(git_info, git_info_file)
      git_info_file.write("\n")

  # Write args to args.log_dir/args.json
  with open(os.path.join(args.log_dir, 'args.json'), 'w') as args_file:
    json.dump(vars(args), args_file)