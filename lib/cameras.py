import numpy as np

class PinholeCamera:

    def __init__(self, calibration=None, pose=None):

        # Get input parameters
        self.calibration = calibration if calibration is not None else np.identity(3)
        self.pose = pose if pose is not None else np.identity(4)
        self.inverse_pose = np.linalg.inv(self.pose)

    def project(self, points_3d):

        # Assuming points column vectors
        points_3d_h = np.ones((points_3d.shape[0] + 1, points_3d.shape[1]))
        points_3d_h[:-1, :] = points_3d
        points_2d_h = np.dot(np.dot(self.calibration, self.inverse_pose[:3,]), points_3d_h)
        points_2d = points_2d_h[:-1, :] / points_2d_h[-1:, :]

        return points_2d