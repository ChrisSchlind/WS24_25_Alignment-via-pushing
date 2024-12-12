import os
import sys
from contextlib import contextmanager
import pybullet as p
from pybullet_utils.bullet_client import BulletClient


@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)


def setup_bullet_client(render: bool = True):
    with stdout_redirected():
        bullet_client = BulletClient(connection_mode=p.GUI)
        bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if not render:
            bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    return bullet_client


def get_link_index(bullet_client, body_id, link_name):
    for i in range(bullet_client.getNumJoints(body_id)):
        if bullet_client.getJointInfo(body_id, i)[12].decode('utf-8') == link_name:
            return i
    raise ValueError(f"Link {link_name} not found in body {body_id}")
