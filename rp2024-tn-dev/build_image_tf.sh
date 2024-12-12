#!/bin/bash

tensorflow_version="2.11.0"
uid=$(eval "id -u")
gid=$(eval "id -g")

docker build \
  --build-arg TENSORFLOW_VERSION="$tensorflow_version" \
  --build-arg UID="$uid" \
  --build-arg GID="$gid" \
  -f tf.Dockerfile \
  -t rp2024/tf:"$tensorflow_version" .
