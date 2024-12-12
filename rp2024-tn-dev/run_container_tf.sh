#!/bin/bash
##############################################################################
##                            Run the container                             ##
##############################################################################
tensorflow_version="2.11.0"

SRC_CONTAINER=/home/jovyan/workspace/src
SRC_HOST="$(pwd)"/src
DATA_CONTAINER=/home/jovyan/data
DATA_HOST="$(pwd)"/data

docker run \
  --name rp2024-tf \
  --privileged \
  --rm \
  -it \
  --net=host \
  -v "$SRC_HOST":"$SRC_CONTAINER":rw \
  -v "$DATA_HOST":"$DATA_CONTAINER":rw \
  -e DISPLAY="$DISPLAY" \
  --gpus all \
 rp2024/tf:"$tensorflow_version"
