from gym import spaces

import os, inspect
import pybullet as p
import numpy as np
import random
from minitouch.env.panda.panda_gym import PandaEnv
from minitouch.env.panda.common.bound_3d import Bound3d

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
urdfRootPath = currentdir + "/assets/"


class PandaHaptics(PandaEnv):
    def __init__(self, delta_step_joint=0.06, delta_step_fingers=0.075, discrete_grasp=False, grasp_threshold=-1,
                 haptics_upper_bound=500, **kwargs):

        super(PandaHaptics, self).__init__(**kwargs)
        # Robots bounds for this tasks
        self.space_limits = Bound3d(0.5, 0.75, -0.20, 0.25, 0.01, 0.025)
        self.max_force = haptics_upper_bound

        if self.grayscale:
            self.visual_observation_space = spaces.Box(low=0, high=255,
                                                       shape=(1, self.height_camera, self.width_camera))
        else:
            self.visual_observation_space = spaces.Box(low=0, high=255,
                                                       shape=(3, self.height_camera, self.width_camera))

        self.vector_observation_space = spaces.Box(low=np.array(
            [0, 0, -self.max_force, -self.max_force, -self.max_force, -self.max_force, -self.max_force, -self.max_force,
             self.space_limits.x_low, self.space_limits.y_low,
             self.space_limits.z_low]),
            high=np.array(
                [0.04, 0.04] + [self.max_force] * 6 + [
                    self.space_limits.x_high, self.space_limits.y_high,
                    self.space_limits.z_high
                ]))

        self.haptics_space = spaces.Box(low=np.array(
            [-self.max_force] * 6),
            high=np.array([self.max_force] * 6)
        )

        self.observation_space = spaces.Tuple((self.visual_observation_space, self.vector_observation_space))

        # Amplitude of each action
        self.delta_step_joint = delta_step_joint
        self.delta_step_fingers = delta_step_fingers
        self.discrete_grasp = discrete_grasp
        self.grasp_threshold = grasp_threshold
        self.state_grasp = False

    def reset(self):
        state = super().reset()
        self.state_grasp = False
        return state

    def step(self, action, grasp = False, min_x=0, max_x=0, min_y=0, max_y=0):

        if self.discrete_grasp:
            if action[3] > (1 - self.grasp_threshold):
                self.state_grasp = True
            elif action[3] < (-1 + self.grasp_threshold):
                self.state_grasp = False

            if self.state_grasp:
                action[3] = -1
            else:
                action[3] = 1

        state, reward, done, info = super().step(action, grasp , min_x, max_x, min_y, max_y)
        return state, reward, done, info

    def set_grasp_properties(self, discrete_grasp, grasp_threshold=0.4):
        self.discrete_grasp = discrete_grasp
        self.grasp_threshold = grasp_threshold

    # We override _get_target_pos to take into accounts the bounds.
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
        target_pos_x = max(self.space_limits.x_low, min(current_end_effector_pos[0] + dx, self.space_limits.x_high))
        target_pos_y = max(self.space_limits.y_low, min(current_end_effector_pos[1] + dy, self.space_limits.y_high))
        target_pos_z = max(self.space_limits.z_low, min(current_end_effector_pos[2] + dz, self.space_limits.z_high))
        return [target_pos_x, target_pos_y, target_pos_z]

    def _debug_step(self):
        super()._debug_step()
        # self.space_limits.pybullet_debug_draw(p, [255, 255, 255])

        # p.addUserDebugText("End effector pos: " + str(np.round(self.get_end_effector_pos(), 2)), [0.3, -0.2, 0.1])
        # p.addUserDebugText("Reward: " + str(round(self._get_reward(), 2)), [0.3, -0.2, 0.2])
        # p.addUserDebugText("Fingers pos: " + str(round(self.get_fingers_pos()[0], 3)), [0.3, -0.2, 0.3])
        # p.addUserDebugText("Left finger force: " + str(np.round(self.get_left_finger_force_vec(), 2)), [0.3, -0.2, 0.5])
        # p.addUserDebugText("Right finger force: " + str(np.round(self.get_right_finger_force_vec(), 2)),
        #                    [0.3, -0.2, 0.6])

        pass

    def _get_reward(self):
        return 0

    def _get_done(self):
        return False

    def _get_haptics(self):
        """

        :return: Haptics state only.
        """
        return np.clip(self.get_left_finger_force_vec() + self.get_right_finger_force_vec(), -self.max_force,
                       self.max_force).tolist()

    def get_vector_state(self):
        """
        :return: Get the vector state returned by the environment
        """
        # TODO: Add angular velocity + linear velocity of end effector
        return self.get_wrist_force_vec()

    def get_state(self):
        """
        Returns state[0]: RGB img (3 x 84 x 84),
                state[1]: Wrench (6 x 1)
        -------

        """
        return self.get_all_sides_image(self.width_camera, self.height_camera), self.get_vector_state()

    def get_all_sides_image(self, width, height):

        self.side_pos_camera = [(self.space_limits.x_high + self.space_limits.x_low) / 2,
                                self.space_limits.y_low,
                                0.1]

        self.side_orn_camera = [0, 0, 0]

        side_image = self.render_image(self.side_pos_camera, self.side_orn_camera, width, height, nearVal=0.1)

        self.back_pos_camera = [self.space_limits.x_low,
                                (self.space_limits.y_high + self.space_limits.y_low) / 2,
                                0.1]

        self.back_orn_camera = [-90, 0, 0]

        back_image = self.render_image(self.back_pos_camera, self.back_orn_camera, width, height, nearVal=0.1)

        self.top_pos_camera = [(self.space_limits.x_high + self.space_limits.x_low) / 2,
                               (self.space_limits.y_high + self.space_limits.y_low) / 2,
                               -0.2]

        #self.top_pos_camera = [(self.space_limits.x_high + self.space_limits.x_low) / 2,
        #                       (self.space_limits.y_high + self.space_limits.y_low) / 2,
        #                        1.0]

        # self.top_pos_camera = [0.7, 0, -0.09]
        self.top_orn_camera = [90, -60, 0]
        #self.top_orn_camera = [90, -60, 180]

        rgb_image, depth_image = self.render_image(self.top_pos_camera, self.top_orn_camera, width, height, nearVal=0.26)
        #import pdb; pdb.set_trace()
        # return np.concatenate((top_image, side_image, back_image), axis=0)
        return rgb_image
        # return np.expand_dims(top_image,0)
