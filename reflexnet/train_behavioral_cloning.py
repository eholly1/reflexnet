import argparse
from behavioral_cloning import BCTrainer, BCFrameDataset
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
  dataset = BCFrameDataset(batch_size, demo_filepath)
  policy = 
  trainer = BCTrainer(
    model=policy,
    dataset=dataset,
    learning_rate=learning_rate,
    )
  trainer.train_and_test(
    log_dir=log_dir,
    train_steps=train_steps,
  )

def main():
  parser = argparse.ArgumentParser()
  parser.AddArgument('--log_dir', required=True, type=str, help='Parent directory under which to save output.')
  parser.AddArgument('--env_name', defailt='Roboschool-Walker2d-v1', type=str, help='Parent directory under which to save output.')
  parser.AddArgument('--demo_filepath', required=True, type=str, help='Full path to file with task demos.')
  parser.AddArgument('--batch_size', default=16, type=int, help='Full path to file with task demos.')
  parser.AddArgument('--learning_rate', default=1e-5, type=int, help='Full path to file with task demos.')
  parser.AddArgument('--train_steps', default=10000, type=int, help='Full path to file with task demos.')
  args = parser.parse_args()

  args.log_dir = os.path.join(args.log_dir, 'behavioral_cloning', args.env_name)

  utils.init_log_dir(args)

  train_behavioral_cloning(**vars(args))

if __name__ == "__main__":
  main()