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

# Set the working directory to /reflexnet_run
WORKDIR /reflexnet_workdir

# Copy the current directory contents into the container at /reflexnet_run
# ADD ./reflexnet /reflexnet_workdir/reflexnet
# ADD ./data /reflexnet_workdir/data

CMD tail -f /dev/null