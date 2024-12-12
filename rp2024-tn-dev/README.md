# Robot programming HKA WS 2024

This repository contains the code for the robot programming course (the policy learning part) at the University of Applied Sciences Karlsruhe.

We will use and update this repository throughout the course. Hello.

## Quick start

### Environment setup

**Requirements:** have docker installed including the post-installation steps.

**Note:** The default settings are for nvidia GPU support. If you don't have an nvidia GPU, open up `build_image.sh` and set the `render` argument to `base`. Also, remove the `--gpus all` flag from the `docker run` command in `run_container.sh`.

Build the docker image with

```bash
./build_image.sh
```

Run the container with
```bash
./run_container.sh
```

Check whether you can open a window from the container by running
```bash
python view_noise_image.py
```
A window should pop up showing a random noise image. You can close it by pressing any key, while the window is focused.

### Basics
Check out the `basics.py` script to get familiar with the API we use and the `Affine` class.

If you want to understand a bit more how everything works under the hood, check out the scripts in the `bullet_env` folder as well as the `transform.py` file.
