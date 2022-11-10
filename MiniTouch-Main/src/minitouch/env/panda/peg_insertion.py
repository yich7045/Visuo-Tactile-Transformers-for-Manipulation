from gym import spaces

import os, inspect
import pybullet as p
import numpy as np
import random
import math
from minitouch.env.panda.panda_haptics import PandaHaptics
from minitouch.env.panda.common.log_specification import LogSpecification
from minitouch.env.panda.common.bound_3d import Bound3d

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
urdfRootPath = currentdir + "/assets/"


class Insertion(PandaHaptics):

    def __init__(self, threshold_found=0.058, cube_spawn_distance=0.1, sparse_reward_scale=1, random_side=False, random_seed=0,
                 **kwargs):
        super(Insertion, self).__init__(**kwargs)
        """
        This task is designed by ourself for VTT.
        """
        self.objectUid = None
        self.object_file_path = os.path.join(urdfRootPath, "objects/Peg/Peg.urdf")
        self.target_file_path = os.path.join(urdfRootPath, "objects/Hole/Hole.urdf")
        self.max_z = 0.2
        self.space_limits = Bound3d(0.2 + 0.2, 0.45 + 0.2, -0.10, 0.10, 0, self.max_z)
        self.action_repeat = 1
        self.sparse_reward_scale = sparse_reward_scale
        self.random_side = random_side

        self.cube_spawn_distance = cube_spawn_distance

        self.target_cube_pos = [(0.4+0.65)/2., 0, 0.03]
        self.random_cube_angle_pos = 0

        if not random_side:
            self.cube_pos_distribution = spaces.Box(
                low=np.array([self.space_limits.x_low + 0.05, self.space_limits.y_low + 0.05, 0.02]),
                high=np.array(
                    [self.space_limits.x_high - 0.05, self.space_limits.y_high - self.cube_spawn_distance, 0.02]))
        else:
            self.cube_pos_distribution = spaces.Box(
                low=np.array([self.space_limits.x_low + self.cube_spawn_distance,
                              self.space_limits.y_low + self.cube_spawn_distance, 0.02]),
                high=np.array([self.space_limits.x_high - self.cube_spawn_distance,
                               self.space_limits.y_high - self.cube_spawn_distance, 0.02]))

        self.log_specifications = [
            LogSpecification("object_distance", "compute_average", 1, "object_distance"),
            LogSpecification("haptics", "compute_variance", 1, "variance_haptics"),
            LogSpecification("cube_pos", "compute_heat_map_x_y", 10, "cube_pos_heatmap", [0.5, 0.95, -0.20, 0.25]),
            LogSpecification("end_effector_pos", "compute_heat_map_x_y", 10, "end_effector_heatmap",
                             [0.5, 0.95, -0.20, 0.25]),
            LogSpecification("cube_pos", "compute_variance", 1, "cube_pos_variance"),
            LogSpecification("found", "compute_or", 1, "found_cube"),
            LogSpecification("target_cube_angle", "compute_average", 1, "target_cube_angle")
        ]
        self.threshold = threshold_found

        random.seed(random_seed)
        np.random.seed(random_seed)
        self.action_space.np_random.seed(random_seed)
        self.cube_pos_distribution.seed(random_seed)
        self.observation_space.seed(random_seed)

    def reset(self):
        state = super().reset()
        self.randomize_hand_pos()
        self.place_objects()
        return state

    def randomize_hand_pos(self):
        # Random position hand
        random_range = 0.2
        x_random_hand_distance = random.uniform((0.4+0.65)/2.-random_range, (0.4+0.65)/2.+random_range)
        y_random_hand_distance = random.uniform(-random_range, random_range)
        random_angle_hand = self.random_cube_angle_pos + math.pi
        init_hand_pos = [x_random_hand_distance, y_random_hand_distance, 0.15]

        self.move_hand_to(init_hand_pos)

    def place_objects(self):
        ang = -np.pi * 0.5
        peg_ori = p.getQuaternionFromEuler([ang, 0, 0])
        self.objectUid = p.loadURDF(self.object_file_path, basePosition=(self.get_end_effector_pos()[0], self.get_end_effector_pos()[1], self.get_end_effector_pos()[2] + 0.03),
                                    baseOrientation=(peg_ori[0], peg_ori[1], peg_ori[2], peg_ori[3]),
                                    globalScaling=1)
        hole_ori = p.getQuaternionFromEuler([ang, 0, 0])
        self.targetUid = p.loadURDF(self.target_file_path, basePosition=self.target_cube_pos,
                                    baseOrientation=(hole_ori[0], hole_ori[1], hole_ori[2], hole_ori[3]),
                                    globalScaling=1, useFixedBase=True)
        p.changeDynamics(self.objectUid, -1, 1, lateralFriction=50, rollingFriction=50, spinningFriction=50,)
        p.changeDynamics(self.objectUid, 0, 1, lateralFriction=0., rollingFriction=0., spinningFriction=0., )
        p.changeVisualShape(self.targetUid, -1,
                            rgbaColor=[0.7, 0.7, 0.7, 1])

        p.changeVisualShape(self.objectUid, -1,
                            rgbaColor=[random.uniform(0, 1), random.uniform(0, 1), random.randint(0, 1), 1])
        p.changeVisualShape(self.objectUid, 0,
                            rgbaColor=[random.uniform(0, 1), random.uniform(0, 1), random.randint(0, 1), 1])
        p.changeVisualShape(self.targetUid, -1,
                            rgbaColor=[random.uniform(0, 1), random.uniform(0, 1), random.randint(0, 1), 1])

    def step(self, action):
        step, reward, done, info = super().step(action)
        pose_state = p.getBasePositionAndOrientation(self.objectUid)
        return step, reward, done, info

    def _get_done(self):
        """
        setting done condition: when peg is tilted too much, it ends trials; when task is finished, it ends trial;
        Returns
        -------
        """
        ori = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.objectUid)[1])
        if abs(ori[0]) > 1.62 or abs(ori[0]) < 1.50:
            return 1
        if abs(ori[2]) > 0.1:
            return 1
        if abs(ori[2]) > 0.1:
            return 1
        elif p.getBasePositionAndOrientation(self.objectUid)[0][2] <= self.threshold and self.get_distance(self.get_object_pos()[0:2], self.target_cube_pos[0:2], dim=2) < self.threshold:
            return 1
        return 0

    def get_object_pos(self):
        return p.getBasePositionAndOrientation(self.objectUid)[0]

    def get_object_distance(self):
        """z
        Get distance between end effector and  object.
        :return: eucledian distance
        """
        return self.get_distance(self.get_object_pos(), self.get_end_effector_pos())

    def _get_reward(self):
        """
        penalty for no peg/hole alignment
        dense reward for putting down the peg
        large sparse reward for finishing the task
        Returns reward
        -------
        """
        if self.get_distance(self.get_object_pos()[0:2], self.target_cube_pos[0:2], dim=2) < self.threshold:
            if p.getBasePositionAndOrientation(self.objectUid)[0][2] <= self.threshold:
                finish_reward = self.sparse_reward_scale
            else:
                finish_reward = 0
            insertion_reward = 1*(self.max_z - p.getBasePositionAndOrientation(self.objectUid)[0][2])
            return finish_reward + insertion_reward
        else:
            align_penalty = 5*(self.threshold - self.get_distance(self.get_object_pos()[0:2], self.target_cube_pos[0:2], dim=2))
            return align_penalty