import unittest

import numpy as np
import tensorflow.keras.backend as K

import lib.losses as losses
import lib.spatial_geometry as spatial_geometry
from lib.cameras import PinholeCamera


# Losses assume the input:
#   y = [ quaternion, translation, shape3D ]
#   y = [ q0,q1,q2,q3, t0,t1,t2, sx0,sy0,sz0,...,sxN,syN,szN ]
#   y.shape = [ 1, 4 + 3 + 3*Points3D]
#
# multiview_reprojection_loss: Require random poses concatenated at the end
#   y_mrl = [ y, quaternion_random_view_0, translation_random_view_0, ..., quaternion_random_view_N, translation_random_view_N ]
#   y_mrl = [ y, qr00,qr10,qr20,qr30, t00,t10,t20, ..., qr0N,qr1N,qr2N,qr3N, t0N,t1N,t2N ]
#   y_mrl.shape = [ 1, 4 + 3 + 3*Points3D + 7*RandomViewa ]


def create_y_true_y_pred():

    y_true = np.array([[1., 0., 0., 0., 0., 0., -30.,
                        -0.98854052, 2.12746976, -3.8310884,
                        -0.41849216, 2.69751813, -3.2610400,
                        1., 0., 0., 0., 0., 0., -40.,
                        1., 0., 0., 0., 0., 0., -20.]])

    y_pred = np.array([[1., 0., 0., 0., 0., 0., -30.,
                        -0.9889736, 2.11025506, -3.81418583,
                        -0.42152029, 2.67770837, -3.24673252,
                        0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0.]])

    return y_true, y_pred

# Tests
class TestLosses(unittest.TestCase):

    def test_xqt(self):
        # Create test samples
        y_true, y_pred = create_y_true_y_pred()

        loss = losses.xqt(
            shape_points=2, beta=1e-2, gamma=1e-3)(K.variable(y_true),
                                                   K.variable(y_pred))

        loss_tf = K.eval(loss)

        mse = lambda x, y: ((x - y)**2).mean(axis=-1)

        loss_np = (1. * mse(y_pred[:, 7:7 + 2 * 3], y_true[:, 7:7 + 2 * 3]) +
        	       1e-2 * mse(y_pred[:, :4], y_true[:, :4]) +
                   1e-3 * mse(y_pred[:, 4:7], y_true[:, 4:7])) / (1. + 1e-2 + 1e-3)

        self.assertAlmostEqual(loss_tf[0], loss_np[0], places=4)


    def test_geometric_alignment(self):
        # Create test samples
        y_true, y_pred = create_y_true_y_pred()

        loss = losses.geometric_alignment(shape_points=2)(K.variable(y_true), K.variable(y_pred))

        loss_tf = K.eval(loss)

        x = np.reshape(y_pred[0, 7:7+3*2], (2, 3)).T
        x_gt = np.reshape(y_true[0, 7:7+3*2], (2, 3)).T

        t = spatial_geometry.quaternion_translation_to_pose(y_pred[0, :4], y_pred[0, 4:7])
        t_gt = spatial_geometry.quaternion_translation_to_pose(y_true[0, :4], y_true[0, 4:7])

        xt = np.dot(t[:3, :3], x) + np.expand_dims(t[:3, 3], axis=1)
        xt_gt = np.dot(t_gt[:3, :3], x_gt) + np.expand_dims(t_gt[:3, 3], axis=1)

        loss_np = np.mean(np.mean(np.abs(xt_gt - xt), axis=0), axis=-1)

        self.assertAlmostEqual(loss_tf[0], loss_np, places=4)


    def test_reprojection(self):
        # Create calibration matrix
        calibration = np.eye(3)

        # Create test samples
        y_true, y_pred = create_y_true_y_pred()


        loss = losses.reprojection(
        	shape_points=2,
        	calibration=calibration)(K.variable(y_true), K.variable(y_pred))

        loss_tf = K.eval(loss)

        x = np.reshape(y_pred[0, 7:7+3*2], (2, 3)).T
        x_gt = np.reshape(y_true[0, 7:7+3*2], (2, 3)).T

        camera = PinholeCamera(
            calibration=calibration,
            pose=spatial_geometry.quaternion_translation_to_pose(y_pred[0, :4], y_pred[0, 4:7]))

        x_2d = camera.project(x)
        x_2d_gt = camera.project(x_gt)

        diff = np.square(x_2d_gt - x_2d)

        loss_np = np.mean(np.mean(diff, axis=-1),axis=-1)

        self.assertAlmostEqual(loss_tf[0], loss_np, places=4)


    def test_multiview_reprojection(self):
        # Create calibration matrix
        calibration = np.eye(3)

        # Create test samples
        y_true, y_pred = create_y_true_y_pred()

        loss = losses.multiview_reprojection(
            shape_points=2,
            calibration=calibration,
            virtual_views=2)(K.variable(y_true), K.variable(y_pred))

        loss_tf = K.eval(loss)

        # Loss numpy
        loss_np = 0.
        x = np.reshape(y_pred[0, 7:7+2*3], (2, 3)).T
        x_gt = np.reshape(y_true[0, 7:7+2*3], (2, 3)).T

        pose_pred = spatial_geometry.quaternion_translation_to_pose(y_pred[0, :4], y_pred[0, 4:7])
        pose_pred_inv = np.linalg.inv(pose_pred)
        pose_gt = spatial_geometry.quaternion_translation_to_pose(y_true[0, :4], y_true[0, 4:7])
        pose_gt_1 = spatial_geometry.quaternion_translation_to_pose(y_true[0, 13:17], y_true[0, 17:20])
        pose_gt_2 = spatial_geometry.quaternion_translation_to_pose(y_true[0, 20:24], y_true[0, 24:27])

        # Projection using view 1
        # 	1. Projective cameras for view 1 non-distorted and distorted
        camera_i = PinholeCamera(calibration=calibration, pose=pose_gt_1)
        camera_d = PinholeCamera(calibration=calibration, pose=np.matmul(pose_gt_1, np.matmul(pose_gt, pose_pred_inv)))

        # 	2. Projected 3D points using non-distorted and distorted cameras
        x_2d_gt = camera_i.project(x_gt)
        x_2d_d = camera_d.project(x)

        # 	3. Reprojection loss using view 1 (shape error)
        diff = np.square(x_2d_gt - x_2d_d)
        loss_np += np.mean(diff)

        # Projection using view 2
        # 	1. Projective cameras for view 2 non-distorted and distorted
        camera_i = PinholeCamera(calibration=calibration, pose=pose_gt_2)
        camera_d = PinholeCamera(calibration=calibration, pose=np.matmul(pose_gt_2, np.matmul(pose_gt, pose_pred_inv)))

        #	2. Projected 3D points using non-distorted and distorted cameras
        x_2d_gt = camera_i.project(x_gt)
        x_2d_d = camera_d.project(x_gt)

        #	3. Reprojection loss using view 2 (shape error)
        diff = np.square(x_2d_gt - x_2d_d)
        loss_np += np.mean(diff)

        self.assertAlmostEqual(loss_tf[0], loss_np, places=4)


if __name__ == '__main__':
    unittest.main()
