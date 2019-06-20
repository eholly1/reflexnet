# reflexnet

```
############################ Docker shortcuts
export DOCKER_NAME=reflexnet
alias docker_cd="cd ~/${DOCKER_NAME}"

# Take down docker.
alias DD="docker kill ${DOCKER_NAME}_container && docker container rm ${DOCKER_NAME}_container"
alias DRM="docker container rm ${DOCKER_NAME}_container"

# Build the image.
alias DB="docker_cd && docker build --tag ${DOCKER_NAME}_image ."

# Run the container.
alias DR="docker run --name ${DOCKER_NAME}_container -d -v ${HOME}/reflexnet/:/reflexnet_workdir -v ${HOME}/data/:/data/ ${DOCKER_NAME}_image"

# Rebuild and run the container.
alias DU="DB && DR"

# Open a terminal in the container.
alias DO="docker exec -it ${DOCKER_NAME}_container bash"
```