##############################################################################
##                                 Base Image                               ##
##############################################################################
ARG TENSORFLOW_VERSION=2.11.0
FROM tensorflow/tensorflow:$TENSORFLOW_VERSION-gpu as tf-base
USER root
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

##############################################################################
##                           Rendering Dependencies                         ##
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
  && rm -rf /var/lib/apt/lists/*# Env vars for the nvidia-container-runtime.

RUN DEBIAN_FRONTEND=noninteractive \
	apt update && \
	apt install -y mesa-utils libgl1-mesa-glx libglu1-mesa-dev freeglut3-dev mesa-common-dev libopencv-dev python3-opencv python3-tk
RUN /usr/bin/python3 -m pip install --upgrade pip

##############################################################################
##                                  User                                    ##
##############################################################################
FROM tf-dependencies as tf-user

# install sudo
RUN apt-get update && apt-get install -y sudo

# Create user
ARG USER=jovyan
ARG PASSWORD=automaton
ARG UID=1000
ARG GID=1000
ENV USER=$USER
RUN groupadd -g $GID $USER \
    && useradd -m -u $UID -g $GID -p "$(openssl passwd -1 $PASSWORD)" \
    --shell $(which bash) $USER -G sudo
RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sudogrp

USER $USER
RUN mkdir -p /home/$USER/workspace/src
RUN mkdir -p /home/$USER/data
WORKDIR /home/$USER/workspace/src
CMD ["bash"]

##############################################################################
##                                 Manipulation Tasks                       ##
##############################################################################
FROM tf-user as dependencies

RUN pip install --no-cache-dir opencv-contrib-python
RUN pip install --no-cache-dir transforms3d tensorflow_addons
RUN pip install --no-cache-dir scipy numpy
RUN pip install --no-cache-dir scikit-learn einops
RUN pip install --no-cache-dir wandb pandas
RUN pip install --no-cache-dir imageio
RUN pip install --no-cache-dir msgpack colortrans
RUN pip install --no-cache-dir fastapi uvicorn
RUN pip install --no-cache-dir tensorflow-graphics
RUN pip install --no-cache-dir loguru
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir hydra-core --upgrade
RUN pip install --no-cache-dir httpx pydantic fastapi uvicorn msgpack
RUN pip install --no-cache-dir ftfy regex

ENV PYTHONPATH=/home/$USER/workspace/src/lib:$PYTHONPATH
