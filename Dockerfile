# Use an official Python runtime as a parent image
FROM python:3

# Install python libraries.
RUN pip3 install tensorflow
RUN pip3 install tensorboardX
RUN pip3 install torch
RUN pip3 install tqdm

# Install roboschool and dependencies.
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN pip3 install roboschool

# Set Up Gcloud.
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get install -y apt-transport-https ca-certificates
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install -y google-cloud-sdk

# Set the working directory to /reflexnet_run
WORKDIR /reflexnet_workdir

# Copy the current directory contents into the container at /reflexnet_run
# ADD ./reflexnet /reflexnet_workdir/reflexnet
# ADD ./data /reflexnet_workdir/data

CMD tail -f /dev/null