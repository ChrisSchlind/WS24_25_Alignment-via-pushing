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
  --name ws2425_avp-tf \
  --privileged \
  --rm \
  -it \
  --net=host \
  -v "$SRC_HOST":"$SRC_CONTAINER":rw \
  -v "$DATA_HOST":"$DATA_CONTAINER":rw \
  -e DISPLAY="$DISPLAY" \
 ws2425_avp/tf:"$tensorflow_version"
