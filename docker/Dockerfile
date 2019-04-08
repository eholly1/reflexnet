# Use an official Python runtime as a parent image
FROM python:3

RUN pip3 install torch

# Set the working directory to /reflexnet_run
WORKDIR /reflexnet_run

# Copy the current directory contents into the container at /reflexnet_run
COPY . /reflexnet_run

# Define environment variable
#ENV NAME World

# Run app.py when the container launches
#CMD bash
CMD tail -f /dev/null