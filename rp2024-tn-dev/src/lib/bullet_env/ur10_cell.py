import copy

import numpy as np
import pybullet as p
import pybullet_data
from collections import namedtuple
from loguru import logger
from transform.affine import Affine

from bullet_env.util import stdout_redirected

JointInfo = namedtuple('JointInfo',
                       ['id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce',
                        'maxVelocity', 'controllable'])


class UR10Cell:
    def __init__(self,
                 bullet_client,
                 urdf_path,
                 workspace_bounds,
                 joint_indices=(0, 6),
                 ee_name='tcp_link'):

        self.bullet_client = bullet_client
        self.urdf_path = urdf_path
        self.workspace_bounds = np.array(workspace_bounds)
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.bullet_client.setGravity(0, 0, -10)
        with stdout_redirected():
            self.robot_id = self.bullet_client.loadURDF(self.urdf_path, [0, 0, 0], [0, 0, 0, 1],
                                                        useFixedBase=True,
                                                        flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        link_index = {}
        for _id in range(p.getNumJoints(self.robot_id)):
            link_name = p.getJointInfo(self.robot_id, _id)[12].decode('UTF-8')
            link_index[link_name] = _id
        self.eef_id = link_index[ee_name]
        logger.debug("End effector id: {}".format(self.eef_id))

        self.target_reached = False
        self.home_position = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.target_joint_positions = copy.deepcopy(self.home_position)
        self.current_sequence = []
        self.joint_indices = joint_indices
        self.controllable_joints = self.setup_joints()
        logger.debug("Controllable joints: {}".format(self.controllable_joints))
        logger.info("Robot loaded")
        self.gripper = RobotiqGripper(bullet_client=bullet_client, robot_id=self.robot_id)


    def setup_joints(self):
        all_joints = []
        active_joints = []
        n_joints = self.bullet_client.getNumJoints(self.robot_id)
        for i in range(n_joints):
            info = list(self.bullet_client.getJointInfo(self.robot_id, i))
            controllable = info[2] != self.bullet_client.JOINT_FIXED
            if controllable:
                active_joints.append(info[0])
                self.bullet_client.setJointMotorControl2(self.robot_id, info[0], p.VELOCITY_CONTROL, targetVelocity=0,
                                                         force=0)
            args = [info[0], info[1].decode("utf-8"), info[3], *info[6:12], controllable]
            info = JointInfo(*args)
            all_joints.append(info)

        controllable_joints = active_joints[self.joint_indices[0]:self.joint_indices[1]]

        return controllable_joints

    def home(self):
        logger.debug("Homing robot")
        for i in range(len(self.controllable_joints)):
            self.bullet_client.resetJointState(self.robot_id, self.controllable_joints[i], self.home_position[i])
        self.target_joint_positions = copy.deepcopy(self.home_position)
        self.target_reached = False
        self.current_sequence = []
        self.bullet_client.stepSimulation()
        eef_pose = self.get_eef_pose()
        self.ptp(eef_pose)

    def step(self,
             gain=0.033,
             velocity=0.15,
             precision=0.0025) -> None:
        """
        Move the robot arm in one simulation step.
        
        Calculate the joint positions in the next step and 
        set is as the target position for the bullet motor controller
        """

        current_position = [self.bullet_client.getJointState(self.robot_id, i)[0] for i in self.controllable_joints]
        current_position = np.array(current_position)
        difference = self.target_joint_positions - current_position

        gains = np.ones(len(self.controllable_joints)) * gain

        max_difference = np.max(np.abs(difference))
        if max_difference < precision:
            if len(self.current_sequence) > 0:
                self.target_joint_positions = self.current_sequence.pop(0)
                self.target_reached = False
            else:
                self.target_reached = True

        next_step = self.current_sequence[0] if len(self.current_sequence) > 0 else self.target_joint_positions
        next_diff = np.array(next_step) - current_position
        max_next_diff = np.max(np.abs(next_diff))

        apply_scaling = max_next_diff > 0.2
        scale = velocity / max_difference if apply_scaling else 1
        velocities = difference * scale

        next_position = current_position + velocities
        self.bullet_client.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.controllable_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=next_position,
            positionGains=gains
        )

    def get_eef_pose(self):
        _p, _o, _, _, position, orientation = p.getLinkState(self.robot_id, self.eef_id, computeForwardKinematics=True)
        return Affine(position, orientation)

    def schedule_sequence(self, sequence):
        logger.debug(f"Scheduling sequence with {len(sequence)} steps")

        self.current_sequence = [self.solve_position_ik(pose) for pose in sequence]
        self.target_joint_positions = self.current_sequence.pop(0)
        self.target_reached = False

    def solve_position_ik(self, pose):
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.eef_id,
            pose.translation, pose.quat,
            lowerLimits=[-2 * np.pi] * 6,
            upperLimits=[2 * np.pi] * 6,
            jointRanges=[4 * np.pi] * 6,
            restPoses=list(self.home_position),
            maxNumIterations=10000,
            residualThreshold=1e-5,
            solver=p.IK_DLS
        )
        robot_joint_positions = joint_positions[self.joint_indices[0]:self.joint_indices[1]]

        return robot_joint_positions

    def ptp(self, pose: Affine):
        self.schedule_sequence([pose])
        self.wait_for_movement()

    def lin(self, pose: Affine):
        logger.debug(f"Linearly interpolating to pose {pose}")
        lin_step_size = 0.05
        current_pose = self.get_eef_pose()
        steps = current_pose.interpolate_to(pose, lin_step_size)
        self.schedule_sequence(steps)
        self.wait_for_movement()

    def wait_for_movement(self, max_steps_per_waypoint=2000):
        n_steps = 0
        while not (n_steps > max_steps_per_waypoint):
            n_steps += 1
            self.bullet_client.stepSimulation()
            self.step()
            if self.target_reached:
                n_steps = 0
                if len(self.current_sequence) == 0:
                    break


class RobotiqGripper:
    def __init__(self, bullet_client, robot_id, mimic_parent_name='finger_joint'):
        self.bullet_client = bullet_client
        self.opened = True
        self.robot_id = robot_id

        joints = []
        n_joints = self.bullet_client.getNumJoints(self.robot_id)
        for i in range(n_joints):
            info = list(self.bullet_client.getJointInfo(self.robot_id, i))
            controllable = info[2] != p.JOINT_FIXED
            args = [info[0], info[1].decode("utf-8"), info[3], *info[6:12], controllable]
            info = JointInfo(*args)
            joints.append(info)
        self.mimic_parent_id = [joint.id for joint in joints if joint.name == mimic_parent_name][0]
        logger.debug(f"Mimic parent id: {self.mimic_parent_id}")

        mimic_children_names = \
            {
                'right_outer_knuckle_joint': 1,
                'left_inner_finger_joint': -1,
                'right_inner_finger_joint': -1,
                'right_inner_knuckle_joint': 1,
                'left_inner_knuckle_joint': 1
            }
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in joints if
                                       joint.name in mimic_children_names}

        self.setup_joints()
        logger.info("Gripper loaded")

    def setup_joints(self):
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self.bullet_client.createConstraint(self.robot_id, self.mimic_parent_id,
                                                    self.robot_id, joint_id,
                                                    jointType=p.JOINT_GEAR,
                                                    jointAxis=[0, 1, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=[0, 0, 0])
            self.bullet_client.changeConstraint(c, gearRatio=multiplier, maxForce=10000, erp=1)

    def close(self):
        logger.debug("Closing gripper")
        max_force = 5
        self.bullet_client.setJointMotorControl2(
            self.robot_id, self.mimic_parent_id, p.VELOCITY_CONTROL, targetVelocity=10.0, force=max_force)
        for _ in range(200):
            self.bullet_client.stepSimulation()

    def open(self):
        logger.debug("Opening gripper")
        _, _, _, _ = self.bullet_client.getJointState(self.robot_id, self.mimic_parent_id)
        self.bullet_client.setJointMotorControl2(self.robot_id, self.mimic_parent_id, p.POSITION_CONTROL,
                                                 targetPosition=0.3, force=50)
        for _ in range(200):
            self.bullet_client.stepSimulation()

    def get_state(self):
        position_parent_joint, _, _, _ = self.bullet_client.getJointState(self.robot_id, self.mimic_parent_id)
        return position_parent_joint
