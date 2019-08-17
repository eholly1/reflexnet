#!/usr/bin/python3

# This shell configures the automatic behavior of experiment instances on GCE.
# Based on example at: https://cloud.google.com/community/tutorials/create-a-self-deleting-virtual-machine

import json
import os
import requests
import subprocess
import sys
import time
import traceback

COPY_INTERVAL = 180.0  # 3min

STDOUT_FNAME = '/tmp/startup_py_stdout.txt'
STDERR_FNAME = '/tmp/startup_py_stderr.txt'
DOCKER_OUT_FNAME = '/tmp/docker_stdout.txt'
DOCKER_ERR_FNAME = '/tmp/docker_stderr.txt'

sys.stdout = open(STDOUT_FNAME, 'w')
sys.stderr = open(STDERR_FNAME, 'w')

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
        if self.raw_name is None or self.gcloud_zone is None:
            return False
        src_fname = os.path.split(src)[-1]
        dst = "gs://reflexes_bucket/experiments/%s/%s" % (self.raw_name, src_fname)
        cmd = "gsutil cp -r %s %s" % (src, dst)
        subprocess.call(cmd, shell=True)
        return True

    def copy_experiment_data(self):
        self.copy("/data/daggr/")

    def __del__(self):
        # If an exception happened, write out the traceback to a file, and copy to
        # experiment folder.
        if self.tb is not None:
            print("\n", self.tb, flush=True)
            
        self.copy(STDOUT_FNAME)
        self.copy(STDERR_FNAME)
        # self.copy(DOCKER_OUT_FNAME)
        # self.copy(DOCKER_ERR_FNAME)
        
        print("\ndeleting ", self.raw_name, self.gcloud_zone, flush=True)
        delete_cmd = "gcloud --quiet compute instances delete %s --zone=%s" % (self.raw_name, self.gcloud_zone)
        self._bash_exec(delete_cmd, should_print_output=False)

    def _bash_exec(self, cmd, should_print_output=True):
        print("$ " + cmd, flush=True)
        self.copy(STDOUT_FNAME)
        
        try:
            pipe = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = str(pipe.stdout)[2:-1], str(pipe.stderr)[2:-1]
            if stdout != "":
                if should_print_output:
                    print(stdout + "\n")
            if pipe.returncode != 0:
                if should_print_output:
                    print(stderr + "\n")
                raise subprocess.CalledProcessError(returncode=pipe.returncode, cmd=cmd, output=stdout, stderr=stderr)

            self.copy(STDOUT_FNAME)
            return stdout
        except subprocess.CalledProcessError as err:
            print(str(err.output) + "\n")
            raise err

    def run(self):

        self.tb = None
        self.raw_name = None
        self.gcloud_zone = None

        try:
            # Get the experiment name and id.
            cmd = "curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google'"
            self.raw_name = self._bash_exec(cmd)
            _, branch_name, exp_name, exp_id_str = self.raw_name.split("-")
            exp_id = int(exp_id_str)
            
            # Get google cloud zone.
            cmd = "curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google'"
            self.gcloud_zone = self._bash_exec(cmd)

            # self._bash_exec("sudo apt update")
            # self._bash_exec("sudo apt -y install docker.io")

            os.chdir("/home/eholly_dev/reflexnet")

            self._bash_exec("snap install core")

            self._bash_exec("docker container rm reflexnet_container")
            self._bash_exec("docker build --tag reflexnet_image .")
            # self._bash_exec("docker run --name reflexnet_container -d -v /home/eholly_dev/reflexnet/:/reflexnet_workdir -v /home/eholly_dev/data/:/data/ reflexnet_image")

            # # Check out the right branch.
            # self._bash_exec("git clone https://github.com/eholly1/reflexnet.git")
            # os.chdir("/reflexnet")
            # self._bash_exec("git pull")
            # self._bash_exec("git checkout %s" % branch_name)

            # # Load the config.
            # config = json.load(open("reflexnet/experiment_config.json"))
            # parameters = get_parameters_for_id(config["parameters"], exp_id)
            # params_ls = []
            # for k, v in parameters.items():
            #     params_ls.extend([k, v])

            # # Start Docker
            # os.mkdir("/data")
            # os.chdir("/reflexnet")

            # build_start = time.time()
            # self._bash_exec("docker build --tag reflexnet_image .", should_print_output=False)
            # print('\ndocker build time: %f\n' % (time.time() - build_start))
            
            # # self._bash_exec("docker run --name reflexnet_container -d -v /reflexnet/:/reflexnet_workdir -v /data/:/data/ reflexnet_image")
            # # self._bash_exec("docker exec -it reflexnet_container bash")

            # # Run the experiment.
            # docker_out = open(DOCKER_OUT_FNAME, "w")
            # docker_err = open(DOCKER_ERR_FNAME, "w")

            # args = [str(x) for x in [
            #     "docker", "run",  "reflexnet_image", "python3",
            #     "reflexnet/%s" % config["script"]
            #     ] + params_ls]
            # print("\n\n$ " + " ".join(args)) # Print out the docker command.
            
            # docker_proc = subprocess.Popen(args) # Run docker command in subprocess.
            # print('\nstarted docker\n')
            
            # time.sleep(20)


        except:
            self.tb = traceback.format_exc()

Supervisor().run()