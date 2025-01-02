#!/bin/bash

tensorflow_version="2.11.0"
image_name="ws2425_avp/rl:${tensorflow_version}"

# Check if image exists
if ! docker image inspect "$image_name" >/dev/null 2>&1; then
    echo "Image not found. Building..."
    ./build_image_rl.sh
fi

# Define container and host directories
SRC_CONTAINER=/home/jovyan/workspace/src
SRC_HOST="$(pwd)/src"
DATA_CONTAINER=/home/jovyan/data
DATA_HOST="$(pwd)/data"

# Allow local connections to the X server
xhost +local:docker

# Run the Docker container 
# ==> for GPU support add "--gpus all" after "-e..."  <--------------------------------------------- GPU  JA / NEIN   !!!!!!!!!!!!!!!!!!!
docker run \
  --name ws2425_avp-rl \
  --privileged \
  --rm \
  -it \
  --net=host \
  -v "$SRC_HOST":"$SRC_CONTAINER":rw \
  -v "$DATA_HOST":"$DATA_CONTAINER":rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e PULSE_SERVER=$PULSE_SERVER \
  --gpus all \
  ws2425_avp/rl:"$tensorflow_version"

# Revoke permissions after the container stops
xhost -local:docker

