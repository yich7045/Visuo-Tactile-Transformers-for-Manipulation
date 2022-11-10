import math


class Bound3d(object):
    def __init__(self, x_low, x_high, y_low, y_high, z_low, z_high):
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high
        self.z_low = z_low
        self.z_high = z_high

    def pybullet_debug_draw(self, p, color):
        # Draw box of the limits bounds
        p.addUserDebugLine([self.x_low, self.y_low, self.z_low],
                           [self.x_high, self.y_low, self.z_low], color)
        p.addUserDebugLine([self.x_low, self.y_low, self.z_low],
                           [self.x_low, self.y_high, self.z_low], color)
        p.addUserDebugLine([self.x_low, self.y_low, self.z_low],
                           [self.x_low, self.y_low, self.z_high], color)
        p.addUserDebugLine([self.x_high, self.y_low, self.z_low],
                           [self.x_high, self.y_low, self.z_high], color)
        p.addUserDebugLine([self.x_high, self.y_low, self.z_low],
                           [self.x_high, self.y_high, self.z_low], color)
        p.addUserDebugLine([self.x_high, self.y_high, self.z_low],
                           [self.x_high, self.y_high, self.z_high], color)
        p.addUserDebugLine([self.x_low, self.y_high, self.z_low],
                           [self.x_low, self.y_high, self.z_high], color)
        p.addUserDebugLine([self.x_high, self.y_high, self.z_low],
                           [self.x_low, self.y_high, self.z_low], color)
        p.addUserDebugLine([self.x_high, self.y_high, self.z_high],
                           [self.x_low, self.y_high, self.z_high], color)
        p.addUserDebugLine([self.x_high, self.y_low, self.z_high],
                           [self.x_high, self.y_high, self.z_high], color)
        p.addUserDebugLine([self.x_low, self.y_low, self.z_high],
                           [self.x_high, self.y_low, self.z_high], color)
        p.addUserDebugLine([self.x_low, self.y_low, self.z_high],
                           [self.x_high, self.y_low, self.z_high], color)
        p.addUserDebugLine([self.x_low, self.y_low, self.z_high],
                           [self.x_low, self.y_high, self.z_high], color)

    def get_max_distance(self):
        return self.get_distance([self.x_low, self.y_low, self.z_low], [self.x_high, self.y_high, self.z_high])

    @staticmethod
    def get_distance(vec, target_vec):
        """

        :param vec: 3d-array of first vector
        :param target_vec: 3d-array of second vector
        :return: Eucledian distance between two 3d vectors.
        """
        return math.sqrt(
            (vec[0] - target_vec[0]) ** 2 + (vec[1] - target_vec[1]) ** 2 + (vec[2] - target_vec[2]) ** 2)

    def is_inside(self, vec):
        return vec[0] > self.x_low and vec[0] < self.x_high and vec[1] > self.y_low and vec[1] < self.y_high