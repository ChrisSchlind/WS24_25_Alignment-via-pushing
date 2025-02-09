#!/bin/bash

tensorflow_version="2.11.0"

# Retrieve the current user's UID and GID
uid=$(id -u)
gid=$(id -g)

# Build the Docker image
docker build \
  --build-arg TENSORFLOW_VERSION="$tensorflow_version" \
  --build-arg UID="$uid" \
  --build-arg GID="$gid" \
  -f rl.Dockerfile \
  -t "ws2425_avp/rl:${tensorflow_version}" .

