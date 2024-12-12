#!/bin/bash

render=base
uid=$(eval "id -u")
gid=$(eval "id -g")

docker build \
  --build-arg RENDER="$render" \
  --build-arg UID="$uid" \
  --build-arg GID="$gid" \
  -f bullet.Dockerfile \
  -t ws2425_avp/bullet .
