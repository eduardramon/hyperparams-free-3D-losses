import numpy as np

# Quaternions
def quaternion_to_rotation_matrix(q):

    R = np.empty((3,3))

    R[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    R[0,1] = 2. * q[1] * q[2] - 2. * q[0] * q[3]
    R[0,2] = 2. * q[1] * q[3] + 2. * q[0] * q[2]
    R[1,0] = 2. * q[1] * q[2] + 2. * q[0] * q[3]
    R[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    R[1,2] = 2. * q[2] * q[3] - 2. * q[0] * q[1]
    R[2,0] = 2. * q[1] * q[3] - 2. * q[0] * q[2]
    R[2,1] = 2. * q[2] * q[3] + 2. * q[0] * q[1]
    R[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    return R

def quaternion_translation_to_pose(quaternion, translation):
    pose = np.identity(4)
    pose[0:3, 0:3] = quaternion_to_rotation_matrix(quaternion)
    pose[0:3, 3] = translation

    return pose