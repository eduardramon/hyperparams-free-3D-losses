import numpy as np

import lib.losses as losses
import tensorflow as tf
import tensorflow.keras as keras

# This is an example on how to plug the losses into your model
# Assumptions
#   - 3DMM defined with:
#       * N 3D points
#       * M basis
#       * mean = [1, 3*N]
#       * eigen_values = [M,1]
#       * eigen_vectors = [M, 3*N]
#
#   - 3DMM coefficients to x and x to 3DMM coefficients
#       * x = m + coeffs · diag(sqrt(eigen_values)) · eigen_vectors
#       * coeffs = (x - m) · eigen_vectors.T · diag(1/sqrt(eigen_values))
#
#   - Camera calibration K = [3,3]
#   - Random views V concatenated at the end of y_true

# Constants
N = 5           # 3D Shape points
M = 2           # Basis
K = np.eye(3,3) # Camera calibration
B = 4           # Batch size
V = 2			# Random views

# Model
#   1. Input layer
i = keras.layers.Input((32, 32, 3)) # Assuming images of 32x32x2

#   2. Processing
x = keras.layers.Flatten()(i)
q = keras.layers.Dense(4)(x)                    # Output for quaternion
t = keras.layers.Dense(3)(x)                    # Output for translation
c = keras.layers.Dense(M)(x)                    # Output for coefficients
x = keras.layers.Dense(3*N, trainable=False)(c) # Output for shape
                                                # Tip: Proper initializaer can plug 3DMM into Dense

#   3. Output layer
o = keras.layers.concatenate( [ q, t, x ], axis=-1 )

#   4. Model
model = keras.models.Model(inputs=i, outputs=o)


# Training
#   1. Compile model
xqt          = losses.xqt(shape_points=N, beta=1.,gamma=1.)
gal 	     = losses.geometric_alignment(shape_points=N)
reprojection = losses.reprojection(shape_points=N, calibration=K)
mrl          = losses.multiview_reprojection(shape_points=N, calibration=K, virtual_views=V)
model.compile( optimizer=keras.optimizers.Adam(), loss=mrl)

#   2. Load data
X = np.random.rand(4*B, 32, 32, 3)
Y = np.random.rand(4*B, 4 + 3 + 3*N + 7*V)

#   3. Fit
model.fit(X, Y, epochs=10, batch_size=B)


#   Evaluation
X = np.random.rand(4*B, 32, 32, 3)
Y = np.random.rand(4*B, 4 + 3 + 3*N + 7*V) # Assuming 4 batches
print(model.evaluate(X,Y, batch_size=B))