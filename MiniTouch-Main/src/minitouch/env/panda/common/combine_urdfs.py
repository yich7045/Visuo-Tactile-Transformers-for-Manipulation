from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed
import pybullet
import pybullet_data
import time
import os


def combine_urdfs(urdf1, urdf2, target):
    """
    :param urdf1: First object.
    :param urdf2: Second object.
    :param target: Position of second object combined.
    :param orn: Orientation of second object combined
    :return: Path of the new urdf.
    """

    p0 = bc.BulletClient(connection_mode=pybullet.DIRECT)
    p1 = bc.BulletClient(connection_mode=pybullet.DIRECT)

    #can also connect using different modes, GUI, SHARED_MEMORY, TCP, UDP, SHARED_MEMORY_SERVER, GUI_SERVER

    object_1 = p1.loadURDF(urdf1, flags=p0.URDF_USE_IMPLICIT_CYLINDER)
    object_2 = p0.loadURDF(urdf2, [0.0, 0, 0])

    ed0 = ed.UrdfEditor()
    ed0.initializeFromBulletBody(object_1, p1._client)
    ed1 = ed.UrdfEditor()
    ed1.initializeFromBulletBody(object_2, p0._client)
    #ed1.saveUrdf("combined.urdf")

    parentLinkIndex = 0

    jointPivotXYZInParent = [0, 0, 0]
    jointPivotRPYInParent = [0, 0, 0]

    jointPivotXYZInChild = target
    jointPivotRPYInChild = [0, 0, 0]

    newjoint = ed0.joinUrdf(ed1, parentLinkIndex, jointPivotXYZInParent, jointPivotRPYInParent,
                            jointPivotXYZInChild, jointPivotRPYInChild, p0._client, p1._client)
    newjoint.joint_type = p0.JOINT_FIXED

    ed0.saveUrdf("/tmp/combined.urdf")

    del p0
    del p1
    del ed0
    del ed1
    return "/tmp/combined.urdf"


