import tensorflow as tf
import tensorflow.keras.backend as K

from .layers import *


# The following losses assume the input:
#   y = [ quaternion, translation, shape3D ]
#   y = [ q0,q1,q2,q3, t0,t1,t2, sx0,sy0,sz0,...,sxN,syN,szN ]
#   y.shape = [ 1, 4 + 3 + 3*Points3D]
#
# multiview_reprojection_loss: Require random poses concatenated at the end
#   y_mrl = [ y, quaternion_random_view_0, translation_random_view_0, ..., quaternion_random_view_N, translation_random_view_N ]
#   y_mrl = [ y, qr00,qr10,qr20,qr30, t00,t10,t20, ..., qr0N,qr1N,qr2N,qr3N, t0N,t1N,t2N ]
#   y_mrl.shape = [ 1, 4 + 3 + 3*Points3D + 7*RandomViewa ]


# Loss = ( alpha * || s - s ||2 + beta * || q - q ||2 + gamma * || p - p ||2 ) / (alpha + beta + gamma)
# NOTE: Alpha is fixed to 1.
def xqt(shape_points, beta, gamma):
    A = K.constant(1.)
    B = K.constant(beta)
    G = K.constant(gamma)
    mse = tf.keras.losses.mean_squared_error

    def xqt_loss(y_true, y_pred):
        return A * mse(y_pred[:, 7:7 + 3 * shape_points], y_true[:, 7:7 + 3 * shape_points]) + \
               B * mse(y_pred[:, :4], y_true[:, :4]) + \
               G * mse(y_pred[:, 4:7], y_true[:, 4:7])

    return xqt_loss


# Loss = Sum_b( Sum_p( || [ R | t ] x - [ R' | t' ] x' || ) / N_p ) / N_b
def geometric_alignment(shape_points):

    f_quaternion_position_to_matrix = QuaternionPositionToMatrixLayer()

    def transform_scenes(inputs):
        t = inputs[0]                                             # Transforms [3,4]
        x = inputs[1]                                             # Scenes [M,3]
        M, D = x.get_shape()
        x_h = tf.concat([x, tf.ones([M, 1], tf.float32)], axis=1) # [M,4]
        xt = tf.transpose(tf.matmul(t, tf.transpose(x_h)))        # [M,3]

        return xt

    def geometric_alignment_loss(y_true, y_pred):
        # Compute 3D points
        x_pred = K.reshape(y_pred[:, 7:7 + shape_points * 3], (-1, shape_points, 3))
        x_gt = K.reshape(y_true[:, 7 :7 + shape_points * 3], (-1, shape_points, 3))

        # Compute pose transforms
        t_pred = f_quaternion_position_to_matrix(y_pred[:, :7])
        t_gt = f_quaternion_position_to_matrix(y_true[:, :7])

        # Transform scenes
        xt_pred = tf.map_fn(transform_scenes, (t_pred, x_pred), dtype=tf.float32)
        xt_gt = tf.map_fn(transform_scenes, (t_gt, x_gt), dtype=tf.float32)

        # Perform average across dimensions and points
        xt_error = tf.reduce_mean(tf.abs(xt_gt - xt_pred), axis=-1)
        error = tf.reduce_mean(xt_error, axis=-1)

        return error

    return geometric_alignment_loss


# Loss = Sum_b( Sum_p( | Proj( x3d ) - Proj'( x3d' ) | ) / N_p ) / N_b
def reprojection(shape_points,
                 calibration):
    f_quaternion_position_to_projection_matrix = QuaternionPositionToProjectionMatrixLayer(calibration)
    f_reprojection_3D_2D = Reprojection3D2DLayer()

    def reprojection_loss(y_true, y_pred):
        # Compute 3D points
        x_pred = K.reshape(y_pred[:, 7:7 + shape_points * 3],(-1, shape_points, 3))
        x_gt = K.reshape(y_true[:, 7:7 + shape_points * 3],(-1, shape_points, 3))

        # Compute projection matrices
        pm_pred = f_quaternion_position_to_projection_matrix(y_pred[:, :7])
        pm_gt = f_quaternion_position_to_projection_matrix(y_true[:, :7])

        # Perform projections
        x_2d_pred = f_reprojection_3D_2D([x_pred, pm_pred])
        x_2d_gt = f_reprojection_3D_2D([x_gt, pm_gt])
        # Perform average across dimensions and points
        x2d_error = tf.reduce_mean(tf.square(x_2d_gt - x_2d_pred), axis=-1)
        error = tf.reduce_mean(x2d_error, axis=-1)

        return error

    return reprojection_loss


# Loss = Sum_b( Sum_v( Sum_p( | Proj_v( x3d ) - Proj'_v( x3d' ) | ) / N_p ) / N_v ) / N_b
# Proj'_v( X ) = K Pv Ptrue Ppred_inv X
# Proj_v( X ) = K Pv X
def multiview_reprojection(shape_points,
                           calibration,
                           virtual_views=2):

    # Methods
    f_quaternion_position_to_matrix = QuaternionPositionToMatrixLayer()
    f_inverse_pose_matrix = InversePoseMatrixLayer()
    f_quaternion_position_to_projection_matrix = QuaternionPositionToProjectionMatrixLayer(calibration)
    f_reprojection_3D_2D = Reprojection3D2DLayer()

    # Custom methods
    def multiply_projection_matrices(inputs):
        pose_mat_true = tf.concat([inputs[0], [[0., 0., 0., 1.]]], axis=0)
        pose_mat_pred_inv = tf.concat([inputs[1], [[0., 0., 0., 1.]]], axis=0)
        projection_v = inputs[2]
        return tf.matmul(projection_v, tf.matmul(pose_mat_true, pose_mat_pred_inv))

    # Constants
    virtual_views_normalization = K.constant(virtual_views, 'float32')

    def multiview_reprojection_loss(y_true, y_pred):

        # Compute x
        x3d_true = K.reshape(y_true[:, 7:7 + shape_points * 3], (-1, shape_points, 3))
        x3d_pred = K.reshape(y_pred[:, 7:7 + shape_points * 3], (-1, shape_points, 3))

        # Init error
        error = 0.

        for i in range(1, virtual_views + 1):

            # Poses
            pose_mat_true = f_quaternion_position_to_matrix(y_true[:, :7])
            pose_mat_pred = f_quaternion_position_to_matrix(y_pred[:, :7])
            pose_mat_pred_inv = f_inverse_pose_matrix(pose_mat_pred)

            # Projection matrices
            projection_v = f_quaternion_position_to_projection_matrix(y_true[:, shape_points * 3 + 7 * i : shape_points * 3 + 7 * (i + 1)])
            projection_true = tf.map_fn(multiply_projection_matrices, (pose_mat_pred, pose_mat_pred_inv, projection_v), dtype=tf.float32)
            projection_pred = tf.map_fn(multiply_projection_matrices, (pose_mat_true, pose_mat_pred_inv, projection_v), dtype=tf.float32)

            # Compute error
            x2d_true = f_reprojection_3D_2D([x3d_true, projection_true])
            x2d_pred = f_reprojection_3D_2D([x3d_pred, projection_pred])
            x2d_error = tf.reduce_mean(tf.square(x2d_true - x2d_pred), axis=-1)
            error += tf.reduce_mean(x2d_error, axis=-1)

        return error / virtual_views_normalization

    return multiview_reprojection_loss
