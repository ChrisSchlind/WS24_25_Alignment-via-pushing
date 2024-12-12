import time
import pybullet as p
from pybullet_utils.bullet_client import BulletClient

from bullet_env.ur10_cell import UR10Cell
from transform.affine import Affine

RENDER = True
URDF_PATH = "/home/jovyan/data/assets/urdf/robot.urdf"

bullet_client = BulletClient(connection_mode=p.GUI)
bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
if not RENDER:
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

robot = UR10Cell(bullet_client=bullet_client, urdf_path=URDF_PATH)


# robot commands 
# move the robot to the home position instantly, without real execution
robot.home()

# get the current end effector pose
home_pose = robot.get_eef_pose()
print(home_pose)

# create a relative pose to the current end effector pose
relative_pose = Affine(translation=[0, 0, 0.1])
# apply the relative pose to the current end effector pose (in the eef frame)
# the multiplication of two Affine objects behaves the same way as the multiplication of two homogeneous transformation matrices
target_pose = home_pose * relative_pose
# move to the target pose
robot.ptp(target_pose)

# per default, the z axis of the eef frame is pointing downwards
# applying the same relative transformation to the target pose in the base frame
current_pose = robot.get_eef_pose()
target_pose = relative_pose * current_pose
# now with linear motion
robot.lin(target_pose)

# open the gripper
robot.gripper.open()
time.sleep(5)

# close the gripper
robot.gripper.close()
time.sleep(5)
# open the gripper
robot.gripper.open()
time.sleep(5)

# close the gripper
robot.gripper.close()
time.sleep(5)
# open the gripper
robot.gripper.open()
time.sleep(5)
# close the gripper
robot.gripper.close()
time.sleep(5)

# open the gripper
robot.gripper.open()
time.sleep(5)
# close the gripper
robot.gripper.close()
time.sleep(5)

# open the gripper
robot.gripper.open()
time.sleep(5)

for _ in range(1000):
    bullet_client.stepSimulation()
    time.sleep(1 / 100)