import gym
from gym import spaces

import os, inspect
import pybullet_data
import pybullet as p
import math
import numpy as np
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
urdfRootPath = currentdir + "/assets/"
import time

class PandaEnv(gym.Env):

    def __init__(self, debug=False, grayscale=False, lf_force=500, rf_force=450):
        if debug:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT, options="--opengl2")


        p.setPhysicsEngineParameter(enableFileCaching=0)

        self.lf_force = lf_force
        self.rf_force = rf_force

        self.episode_step_counter = 0
        self.episode_counter = 0
        self.pandaUid = None
        self.tableUid = None
        self.init_panda_joint_state = [-0.028, 0.853, -0.016, -1.547, 0.017, 2.4, 2.305, 0., 0.]

        self.wrist_idx = 7
        self.left_finger_idx = 9
        self.right_finger_idx = 10
        self.end_effector_idx = 11

        self.debug = debug
        self.grayscale = grayscale

        self.ik_precision_treshold = 1e-4
        # set maximum inverse kinematics iterations to be 100
        self.max_ik_repeat = 100

        # Robot always face that direction
        self.fixed_orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])

        # Amplitude of each action
        self.delta_step_joint = 0.016
        self.delta_step_fingers = 0.015

        self.close_gripper = False
        self.start_state = []

        # Render camera setting
        self.height_camera = 84
        self.width_camera = 84

        self.log_specifications = []

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))

    def reset(self):
        """
        Gym reset episode function.
        :return: state
        """

        self.episode_counter += 1
        self.episode_step_counter = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything

        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF('plane.urdf', basePosition=[0.5, 0, -0.65])
        for i in range(len(self.init_panda_joint_state)):
            p.resetJointState(self.pandaUid, i, self.init_panda_joint_state[i])

        p.enableJointForceTorqueSensor(self.pandaUid, self.wrist_idx)
        p.enableJointForceTorqueSensor(self.pandaUid, self.end_effector_idx)
        p.enableJointForceTorqueSensor(self.pandaUid, self.left_finger_idx)
        p.enableJointForceTorqueSensor(self.pandaUid, self.right_finger_idx)

        self.tableUid = p.loadURDF(os.path.join(urdfRootPath, "objects/table/table.urdf"), basePosition=[0.5, 0, -0.65])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return self.get_state()

    def step(self, action, grasp = False, min_x=0, max_x=0, min_y=0, max_y=0):
        """
        Gym simulation.
        :param action: numpy vector containing to action to simulate.
        :return: state, action, done, info
        """
        self.episode_step_counter += 1
        # Enable smooth motion of the robot arm
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        target_end_effector_pos = self._get_target_pos(action)
        if grasp:
            target_end_effector_pos[0:2] = [(min_x + max_x)/2, (min_y + max_y)/2]
        fingers = self.get_fingers_pos()[0] + action[3] * self.delta_step_fingers
        distance = math.inf

        repeat_counter = 0

        # Ensure good action distance + save computation cost + make delta_step customizable without additional tuning
        while distance > self.ik_precision_treshold and repeat_counter < self.max_ik_repeat:

            #if self.debug:
            #    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

            computed_ik_joint_pos = p.calculateInverseKinematics(self.pandaUid, 11, target_end_effector_pos,
                                                                 self.fixed_orientation)

            p.setJointMotorControlArray(self.pandaUid, list(range(7)), p.POSITION_CONTROL,
                                        list(computed_ik_joint_pos[:-2]), forces=[500.0]*7)
            p.setJointMotorControl2(self.pandaUid, self.right_finger_idx,
                                    p.POSITION_CONTROL, fingers, force=self.rf_force)
            p.setJointMotorControl2(self.pandaUid, self.left_finger_idx,
                                    p.POSITION_CONTROL, fingers, force=self.lf_force)
            #p.setJointMotorControl2(self.pandaUid, self.right_finger_idx,
            #                        p.POSITION_CONTROL, fingers, force=20.0)
            #p.setJointMotorControl2(self.pandaUid, self.left_finger_idx,
            #                        p.POSITION_CONTROL, fingers, force=25.0)
            p.stepSimulation()
            #time.sleep(self.fixedTimeStep)

            distance = self.get_distance(target_end_effector_pos, self.get_end_effector_pos())
            repeat_counter += 1

        if self.debug:
            self._debug_step()

        return self.get_state(), self._get_reward(), self._get_done(), self._get_info()

    def simulate(self, action, grasp = False, min_x=0, max_x=0, min_y=0, max_y=0):
        """
        Gym simulation.
        :param action: numpy vector containing to action to simulate.
        :return: state, action, done, info
        """
        self.episode_step_counter += 1
        # Enable smooth motion of the robot arm
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        target_end_effector_pos = self._get_target_pos(action)
        if grasp:
            target_end_effector_pos[0:2] = [(min_x + max_x)/2., (min_y + max_y)/2.]
        fingers = self.get_fingers_pos()[0] + action[3] * self.delta_step_fingers
        distance = math.inf

        repeat_counter = 0
        # Ensure good action distance + save computation cost + make delta_step customizable without additional tuning
        while distance > self.ik_precision_treshold and repeat_counter < self.max_ik_repeat:

            computed_ik_joint_pos = p.calculateInverseKinematics(self.pandaUid, 11, target_end_effector_pos,
                                                                 self.fixed_orientation)

            p.setJointMotorControlArray(self.pandaUid, list(range(7)), p.POSITION_CONTROL,
                                        list(computed_ik_joint_pos[:-2]), forces=[500.0]*7)
            p.setJointMotorControl2(self.pandaUid, self.right_finger_idx,
                                    p.POSITION_CONTROL, fingers, force=self.rf_force)
            p.setJointMotorControl2(self.pandaUid, self.left_finger_idx,
                                    p.POSITION_CONTROL, fingers, force=self.lf_force)

            p.stepSimulation()

            distance = self.get_distance(target_end_effector_pos, self.get_end_effector_pos())
            repeat_counter += 1

        if self.debug:
            self._debug_step()

    def move_hand_to(self, vec):
        self.episode_step_counter += 1
        # Enable smooth motion of the robot arm
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        target_end_effector_pos = vec
        distance = math.inf

        repeat_counter = 0

        # Ensure good action distance + save computation cost + make delta_step customizable without additional tuning
        while distance > self.ik_precision_treshold and repeat_counter < self.max_ik_repeat:

            computed_ik_joint_pos = p.calculateInverseKinematics(self.pandaUid, 11, target_end_effector_pos,
                                                                 self.fixed_orientation)

            p.setJointMotorControlArray(self.pandaUid, list(range(7)), p.POSITION_CONTROL,
                                        list(computed_ik_joint_pos[:-2]))

            p.stepSimulation()
            distance = self.get_distance(target_end_effector_pos, self.get_end_effector_pos())
            repeat_counter += 1

    def _get_target_pos(self, action):
        """
            Give the target position given the action. This is put in a function to be able to modify how action are
            applied for different tasks.
        :param action: Raw action from the user.
        :return: 3d-array of the X, Y, Z target end effector position.
        """
        dx = action[0] * self.delta_step_joint
        dy = action[1] * self.delta_step_joint
        dz = action[2] * self.delta_step_joint
        current_end_effector_pos = self.get_end_effector_pos()
        return [current_end_effector_pos[0] + dx, current_end_effector_pos[1] + dy, current_end_effector_pos[2] + dz]

    def get_left_finger_force_vec(self):
        """
        :return: Force torque 3d vector of the left finger.
        """
        return p.getJointState(self.pandaUid, self.left_finger_idx)[2][:3]

    def get_wrist_force_vec(self):
        # get wrist reaction/torque
        return p.getJointState(self.pandaUid, self.wrist_idx)[2]

    def get_right_finger_force_vec(self):
        """
        :return: Force torque 3d vector of the right finger.
        """
        return p.getJointState(self.pandaUid, self.right_finger_idx)[2][:3]

    def get_vector_state(self):
        """
        :return: Get the vector state returned by the environment
        """
        return self.get_fingers_pos() + self.get_left_finger_force_vec() + \
               self.get_right_finger_force_vec()

    def get_state(self):
        """
        Make it overridable for inheritance. (ex: include visual)
        :return: State.
        """
        return self.get_vector_state()

    def get_end_effector_pos(self):
        """
        :return: The end effector X, Y, Z positions.
        """
        return p.getLinkState(self.pandaUid, self.end_effector_idx)[0]

    def get_fingers_pos(self):
        """
        :return: Tuple (2-dim) of gripper's fingers positions.
        """
        return (p.getJointState(self.pandaUid, self.left_finger_idx)[0],
                p.getJointState(self.pandaUid, self.right_finger_idx)[0])

    def get_all_joint_pos(self):
        """
        :return: Vector of the positions of all the joints of the robot.
        """
        joints_pos = []
        for i in range(len(self.init_panda_joint_state)):
            joints_pos.append(p.getJointState(self.pandaUid, i)[0])
        return joints_pos

    def render_image(self, camera_pos, camera_orn, camera_width, camera_height, nearVal=0.01):
        """
        :param camera_pos:
        :param camera_orn:
        :param camera_width:
        :param camera_height:
        :return:
        """
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_pos,
                                                          distance=.7,
                                                          yaw=camera_orn[0],
                                                          pitch=camera_orn[1],
                                                          roll=camera_orn[2],
                                                          upAxisIndex=2)

        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(camera_width) / camera_height,
                                                   nearVal=nearVal,
                                                   farVal=100.0)

        (_, _, px, depth_image, _) = p.getCameraImage(width=camera_width,
                                            height=camera_height,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                            flags=p.ER_NO_SEGMENTATION_MASK)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (camera_height, camera_width, 4))
        rgb_array = rgb_array[:, :, :3]

        if self.grayscale:
            rgb_array = rgb_array[:, :, 0] * 0.2989 + rgb_array[:, :, 1] * 0.587 + \
                        rgb_array[:, :, 2] * 0.114
            rgb_array = rgb_array[np.newaxis, :, :]
        else:
            rgb_array = np.moveaxis(rgb_array, [0, 1, 2], [1, 2, 0])

        return rgb_array, depth_image

    def _get_reward(self):
        """
        To implement in implemented class.
        :return:
        """
        pass

    def _get_info(self):
        """
        To implement in implemented class.
        :return:
        """
        pass

    def _get_done(self):
        """
        To implement in implemented class.
        :return:
        """
        pass

    def close(self):
        pass

    def _debug_step(self):
        """
        Add debug code here.
        :return:
        """
        p.removeAllUserDebugItems()

    @staticmethod
    def get_distance(vec, target_vec, dim = 3):
        """
        :param vec: 3d-array of first vector
        :param target_vec: 3d-array of second vector
        :return: Eucledian distance between two 3d vectors.
        """
        if dim == 3:
            return math.sqrt((vec[0] - target_vec[0]) ** 2 + (vec[1] - target_vec[1]) ** 2 + (vec[2] - target_vec[2]) ** 2)
        else:
            return math.sqrt((vec[0] - target_vec[0]) ** 2 + (vec[1] - target_vec[1]) ** 2)

