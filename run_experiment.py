import argparse
import json
import subprocess


VALID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"

def _check_name(name):
    for c in name:
        if not c in VALID_CHARS:
            raise ValueError(
                "Found invalid char '%s' in experiment name. Valid chars are (A-Z)(a-z)-" % c)

def main():
    config = json.load(open("reflexnet/experiment_config.json"))

    name = config["name"]
    name = name.replace("_", "-")
    _check_name(name)

    # TODO(eholly): Get branch name from git, modify this field, and possibly recommit.
    branch_name = config["branch_name"]

    num_experiment_jobs = 1
    for param, param_settings in config["parameters"].items():
        if type(param_settings) is list:
            num_experiment_jobs *= len(param_settings)

    for i in range(num_experiment_jobs):
        subprocess.call([
            "gcloud", "compute", "instances", "create", "experiment-%s-%d" % (name, i),
            "--source-instance-template", "experiment-template", "--zone", "us-west1-a",
            "--metadata", "branch_name=%s" % branch_name])


if __name__ == "__main__":
    main()