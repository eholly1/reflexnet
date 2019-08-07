#!/usr/bin/env python3

# This shell configures the automatic behavior of experiment instances on GCE.
# Based on example at: https://cloud.google.com/community/tutorials/create-a-self-deleting-virtual-machine

import json
import requests
import subprocess
import time
import traceback

COPY_INTERVAL = 180.0  # 3min

def get_parameters_for_id(param_config, exp_id):
    parameters = {}

    # Copy out all non-list params.
    for k, v in param_config.items():
        if type(v) != list:
            parameters[k] = v
    for k, _ in parameters.items():
        del param_config[k]            

    # Figure out which setting corresponds to exp_id.
    keys = [k for k in param_config.keys()]
    keys.sort()
    mag = 1
    keys_and_mags = []
    for key in keys:
        keys_and_mags.append((key, mag))
        mag *= len(param_config[key])
    keys_and_mags.reverse()
    for key, mag in keys_and_mags:
        i = 0
        while exp_id >= mag:
            exp_id -= mag
            i += 1
        parameters[key] = param_config[key][i]

    return parameters

class Supervisor:

    def copy(self, src):
        dst = "gs://reflexes_bucket/experiments/%s" % self.raw_name
        cmd = "gsutil cp -r %s %s" % (src, dst)
        subprocess.call(cmd.split(" "))

    def copy_experiment_data(self):
        self.copy("/data/daggr/")

    def __del__(self):
        # If an exception happened, write out the traceback to a file, and copy to
        # experiment folder.
        if self.tb is not None:
            with open("supervisor_error.txt", "w") as f:
                f.write(self.tb)
            self.copy("supervisor_error.txt")

        delete_cmd = "gcloud --quiet compute instances delete %s --zone=%s" % (self.raw_name, self.gcloud_zone)
        subprocess.call(delete_cmd.split(" "))

    def run(self):
        self.tb = None
        try:
            # _________________ Start the experiment.

            # Get the experiment name and id.
            self.raw_name = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/name")
            second_part = self.raw_name.split("-", 1)[1]
            exp_name, exp_id_str = second_part.rsplit("-", 1)
            exp_id = int(exp_id_str)

            # Get the branch name.
            branch_name = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/branch_name")

            # Get google cloud zone.
            self.gcloud_zone = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/zone")

            # Check out the right branch.
            assert not subprocess.call(["cd", "~/reflexnet"])
            assert not subprocess.call(["git", "pull"])
            assert not subprocess.call(["git", "checkout", branch_name])

            # Load the config.
            config = json.load("reflexnet/experiment_config.json")
            parameters = get_parameters_for_id(config["parameters"], exp_id)

            # Build the command line string.
            cmd = "python3 reflexnet/%s" % config["script"]
            for k, v in parameters:
                cmd += " %s %s" % (k, v)

            # Start Docker
            assert not subprocess.call(["DB"])
            docker_proc = subprocess.Popen(["docker", "run" "reflexnet_image"] + cmd.split(" "))

            time.sleep(10.0)

            # _________________ Check that it is running and wait until it finishes. 
            # _________________ Every 3 minutes, copy output to filestorage.


            next_copy_time = time.time()
            while True:
                now = time.time()
                if now > next_copy_time:
                    self.copy_experiment_data()
                    next_copy_time = now + COPY_INTERVAL

                running_python_commands = str(subprocess.check_output(["pgrep", "-af", "python"]))
                if not "reflexnet" in running_python_commands:
                    break

            self.copy_experiment_data()

        except:
            self.tb = traceback.format_exc()

Supervisor().run()