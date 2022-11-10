from gym import spaces
from gym import spaces

import os, inspect
import pybullet as p
import numpy as np
import random
from minitouch.env.panda.panda_haptics import PandaHaptics
from minitouch.env.panda.common.log_specification import LogSpecification
from minitouch.env.panda.common.bound_3d import Bound3d
#from gibson2.objects.articulated_object import ArticulatedObject
#import gibson2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
urdfRootPath = currentdir + "/assets/"
import math

class DoorEnv(PandaHaptics):

    def __init__(self, threshold_found=0.7, minimum_distance_target=0.25, delta_orn=0.3, random_seed=0, **kwargs):
        super(DoorEnv, self).__init__(**kwargs)
        # Robots bounds for this tasks
        self.objectUid = None
        self.action_repeat = 1
        self.object_file_path = os.path.join(urdfRootPath, "objects/cabinet/cabinet_0004.urdf")

        self.space_limits = Bound3d(0.5, 0.75, -0.2, 0.2, 0.1, 0.15)
        self.based_fixed_orientation = [0., math.pi/2, 0]
        self.init_panda_joint_state = [-0.77, 1.50, 0.98, -2.18, 2.83, 1.66, -0.25, 0.0, 0.0]
        self.object_random_scale_range = (0.5, 0.5)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,))

        self.object_start_position = None
        self.treshold_found = threshold_found
        self.minimum_distance_target = minimum_distance_target
        self.current_orn = 0
        self.delta_orn = delta_orn

        self.cube_pos_distribution = spaces.Box(
            low=np.array([self.space_limits.x_low + 0.05, self.space_limits.y_low + 0.05, 0.02]),
            high=np.array([self.space_limits.x_high - 0.05, self.space_limits.y_high - 0.05, 0.02]))

        self.log_specifications = [
            LogSpecification("haptics", "compute_variance", 1, "variance_haptics"),
            LogSpecification("door_angle", "compute_variance", 1, "door_angle_variance"),
            LogSpecification("end_effector_pos", "compute_heat_map_x_y", 10, "end_effector_heatmap",
                             [0.5, 0.95, -0.20, 0.25]),
            LogSpecification("found", "compute_or", 1, "success"),
        ]

        random.seed(random_seed)
        np.random.seed(random_seed)
        self.action_space.np_random.seed(random_seed)
        self.cube_pos_distribution.seed(random_seed)
        self.observation_space.seed(random_seed)

    def reset(self, random_pos=True):
        """
        Parameters
        ----------
        random_pos: Simple curriculum learning: during the training, the door is opened at a random angle (random_pos=True)
                    the door is completely closed during the evaluation (random_pose=False)
        Returns: get initial state Img, Tactile
        -------
        """
        self.fixed_orientation = p.getQuaternionFromEuler((0, self.based_fixed_orientation[1], self.based_fixed_orientation[2]))
        state = super().reset()
        self.object_start_position = list(self.cube_pos_distribution.sample())
        self.place_objects(random_pos)
        return state

    def place_objects(self, random_pos=True):

        self.door = p.loadURDF(self.object_file_path, basePosition=[0.86, 0, 0.45],
                                    globalScaling=1, baseOrientation=[0, 180, 0, 1])
        if random_pos:
            init_door_joint_state = [0, 0.6*random.uniform(0, 1)]
        else:
            init_door_joint_state = [0, 0]
        for i in range(len(init_door_joint_state)):
            p.resetJointState(self.door, i, init_door_joint_state[i])

        p.changeVisualShape(self.door, 0,
                        rgbaColor=[random.uniform(0, 1), random.uniform(0, 1), random.randint(0, 1), 1])
        p.changeVisualShape(self.door, 1,
                            rgbaColor=[random.uniform(0, 1), random.uniform(0, 1), random.randint(0, 1), 1])

    def step(self, action):
        self.current_orn += self.delta_orn * action[4]
        self.current_orn = min(max(self.current_orn, -math.pi/4), math.pi/4)
        self.fixed_orientation = p.getQuaternionFromEuler((self.current_orn, self.based_fixed_orientation[1], self.based_fixed_orientation[2]))
        step, reward, done, info = super().step(action)

        return step, reward, done, info

    def _get_done(self):
        return self._get_door_angle() > self.treshold_found

    def get_object_pos(self):
        return [0, 0, 0, 0, 0, 0]

    def get_object_distance(self):
        """
        Get distance between end effector and  object.
        :return: eucledian distance
        """
        return self.get_distance(self.get_object_pos(), self.get_end_effector_pos())

    def _get_info(self):
        found = self._get_door_angle() > self.treshold_found

        return {"haptics": self._get_haptics(), "fingers_pos": self.get_fingers_pos(),
                "end_effector_pos": self.get_end_effector_pos(), "door_angle": self._get_door_angle(), "found": found}

    def _get_door_joint_pos(self):
        joint_positions = []
        for i in range(p.getNumJoints(self.door)):
            joint_positions.append(p.getJointState(self.door, i)[0])
        return joint_positions

    def _get_door_angle(self):
        return self._get_door_joint_pos()[1]

    def _get_reward(self):
        if self._get_door_angle() > self.treshold_found:
            return 25
        else:
            open_reward = self._get_door_angle() - self.treshold_found
            return 10 * open_reward

    def get_all_sides_image(self, width, height):

        self.top_pos_camera = [(self.space_limits.x_high + self.space_limits.x_low)/2,
                               (self.space_limits.y_high + self.space_limits.y_low)/2,
                               0.05]

        self.top_orn_camera = [0, -90, 0]

        top_image = self.render_image(self.top_pos_camera, self.top_orn_camera, width, height, nearVal=0.01)
        return top_image


class DoorEnvContinuous(DoorEnv):

    def _get_reward(self):
        return max(self._get_door_angle() / self.treshold_found, 0)

    def _get_done(self):
        return self._get_door_angle() > self.treshold_found
