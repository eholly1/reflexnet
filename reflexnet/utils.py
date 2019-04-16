
import json
import os
import subprocess
import tensorflow as tf

import RoboschoolWalker2d_v1_2017jul

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

def tree_apply(fn, tree_node):
  if isinstance(tree_node, dict):
    return {k: tree_apply(fn, tree_node[k]) for k in tree_node}
  if isinstance(tree_node, list):
    return [tree_apply(fn, elem) for elem in tree_node]
  if isinstance(tree_node, tuple):
    return tuple([tree_apply(fn, elem) for elem in tree_node])
  return fn(tree_node)

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