##############################################################################
##                                Base Image                                ##
##############################################################################
ARG TENSORFLOW_VERSION=2.11.0

# CPU or GPU:
#FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}
FROM tensorflow/tensorflow:$TENSORFLOW_VERSION-gpu AS tf-base

USER root
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

##############################################################################
##                           System Dependencies                            ##
##############################################################################
FROM tf-base as tf-dependencies
USER root

RUN apt update \
  && apt install -y -qq --no-install-recommends \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1 \
  libxext6 \
  libx11-6 \
  mesa-utils \
  libgl1-mesa-glx \
  libglu1-mesa-dev \
  freeglut3-dev \
  mesa-common-dev \
  libopencv-dev \
  python3-opencv \
  python3-tk \
  sudo \
  x11-apps \
  && rm -rf /var/lib/apt/lists/*

##############################################################################
##                                User Setup                                ##
##############################################################################
FROM tf-dependencies as tf-user

# install sudo
RUN apt-get update && apt-get install -y sudo

ARG USER=jovyan
ARG PASSWORD=automaton
ARG UID=1000
ARG GID=1000
ENV USER=$USER

RUN groupadd -g $GID $USER \
  && useradd -m -u $UID -g $GID -p "$(openssl passwd -1 $PASSWORD)" \
  --shell $(which bash) $USER -G sudo \
  && echo "%sudo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sudogrp

USER $USER
RUN mkdir -p /home/$USER/workspace/src /home/$USER/data

##############################################################################
##                           Python Dependencies                            ##
##############################################################################
FROM tf-user as dependencies

RUN /usr/bin/python3 -m pip install --upgrade pip

# Combined dependencies from both files
RUN pip install --no-cache-dir \
  numpy \
  scipy \
  loguru \
  h5py \
  hydra-core \
  omegaconf \
  pybullet \
  opencv-python \
  opencv-contrib-python \
  tensorflow-addons \
  scikit-learn \
  einops \
  wandb \
  pandas \
  imageio \
  msgpack \
  colortrans \
  fastapi \
  uvicorn \
  tensorflow-graphics \
  matplotlib \
  ftfy \
  regex

ENV PYTHONPATH=/home/$USER/workspace/src/lib:$PYTHONPATH

##############################################################################
##                              Work Directory                              ##
##############################################################################
WORKDIR /home/$USER/workspace/src
CMD ["bash"]
