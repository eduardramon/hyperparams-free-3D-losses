import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


# Layer implementation for transforming poses [ q, x ] into 3x4 matrices
#   > X : Input       ( Pose 1x7 ) as [ q, p ]
#   > Y : Output      ( Pose 3x4 )
class QuaternionPositionToMatrixLayer(Layer):

    def __init__(self, **kwargs):
        super(QuaternionPositionToMatrixLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(QuaternionPositionToMatrixLayer, self).build(input_shape)

    def call(self, x):
        return tf.map_fn(self._pose_matrix, x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3, 4)

    def _pose_matrix(self, p):
        q = p[:4]
        x = p[4:]

        r = [
            K.square(q[0]) + K.square(q[1]) - K.square(q[2]) - K.square(q[3]),
            2. * q[1] * q[2] - 2. * q[0] * q[3],
            2. * q[1] * q[3] + 2. * q[0] * q[2],
            2. * q[1] * q[2] + 2. * q[0] * q[3],
            K.square(q[0]) - K.square(q[1]) + K.square(q[2]) - K.square(q[3]),
            2. * q[2] * q[3] - 2. * q[0] * q[1],
            2. * q[1] * q[3] - 2. * q[0] * q[2],
            2. * q[2] * q[3] + 2. * q[0] * q[1],
            K.square(q[0]) - K.square(q[1]) - K.square(q[2]) + K.square(q[3])
        ]

        r = K.reshape(K.cast(K.stack(r), 'float32'), (3, 3))
        t = tf.reshape(x, (3, 1))
        pose = tf.concat([r, t], axis=1)

        return pose


# Layer implementation for inverting pose matrices
#   > X : Input       ( Pose 3x4 )
#   > Y : Output      ( Iverse Pose 3x4 )
class InversePoseMatrixLayer(Layer):

    def __init__(self, **kwargs):
        super(InversePoseMatrixLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InversePoseMatrixLayer, self).build(input_shape)

    def call(self, x):
        return tf.map_fn(self._inverse_pose_matrix, x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _inverse_pose_matrix(self, p):
        r = tf.transpose(p[:3, :3])
        t = tf.negative(tf.matmul(r, tf.reshape(p[:3, 3], (3, 1))))
        inv_pose = tf.concat([r, t], axis=1)

        return inv_pose


# Layer implementation for transforming poses [ q, x ] into projection matrices 3x4 matrices
#   > X : Input       ( Pose 1x7 ) as [ q, p ]
#   > Y : Output      ( Projection Matrix ) as [ 3 x 4 ]: Proj Matrix = K · P
#  NOTE: This layer is only implemented for performance purposes
class QuaternionPositionToProjectionMatrixLayer(Layer):

    def __init__(self, calibration, **kwargs):
        self.calibration = tf.convert_to_tensor(calibration, dtype=tf.float32)
        super(QuaternionPositionToProjectionMatrixLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(QuaternionPositionToProjectionMatrixLayer, self).build(input_shape)

    def call(self, x):
        return tf.map_fn(self._projection_matrix, x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3, 4)

    def _projection_matrix(self, x):
        q = x[:4]
        x = x[4:]

        r = [
            K.square(q[0]) + K.square(q[1]) - K.square(q[2]) - K.square(q[3]),
            2. * q[1] * q[2] - 2. * q[0] * q[3],
            2. * q[1] * q[3] + 2. * q[0] * q[2],
            2. * q[1] * q[2] + 2. * q[0] * q[3],
            K.square(q[0]) - K.square(q[1]) + K.square(q[2]) - K.square(q[3]),
            2. * q[2] * q[3] - 2. * q[0] * q[1],
            2. * q[1] * q[3] - 2. * q[0] * q[2],
            2. * q[2] * q[3] + 2. * q[0] * q[1],
            K.square(q[0]) - K.square(q[1]) - K.square(q[2]) + K.square(q[3])
        ]

        rt = tf.transpose(K.reshape(K.cast(K.stack(r), 'float32'), (3, 3)))
        t = tf.negative(tf.matmul(rt, tf.reshape(x, (3, 1))))
        inv_pose = tf.concat([rt, t], axis=1)

        return K.dot(self.calibration, inv_pose)


# Layer implementation for reprojecting a set of points from 3D to 2D assuming a Pinhole Camera
#   > X : Input       ( Nodes, Positions )
#   > P : Camera Pose ( Pose ) = [ q( 4 ), x( 3 ) ]
#   > Y : Output      ( Nodes, Reprojected positions )
class Reprojection3D2DLayer(Layer):

    def __init__(self, **kwargs):
        super(Reprojection3D2DLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Reprojection3D2DLayer, self).build(input_shape)

    def call(self, inputs):
        # Input tensors
        x = inputs[0]  # Shapes
        c = inputs[1]  # Projection matrices

        # Input shapes
        N, M, D = x.get_shape()
        N, Cr, Cc = c.get_shape()

        # Project points
        xc = tf.concat([tf.reshape(x, [-1, M * D]), tf.reshape(c, [-1, Cr * Cc])], axis=1)
        x2d = tf.map_fn(self._project_points(M, D, Cr, Cc), xc)

        return x2d

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape[-1] = output_shape[-1] - 1
        return tuple(output_shape)

    def _project_points(self, M, D, Cr, Cc):

        def project(xc):
            # Recover scene and camera
            x = tf.reshape(xc[:M * D], [M, D])
            c = tf.reshape(xc[M * D:], [Cr, Cc])

            # Project scene
            xh = tf.concat([x, tf.ones((M, 1))], axis=1)
            xp = tf.matmul(xh, tf.transpose(c))
            x2d = xp[:, :2] / tf.expand_dims(xp[:, 2], 1)

            return x2d

        return project