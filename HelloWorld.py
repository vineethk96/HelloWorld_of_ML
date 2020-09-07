import tensorflow as tf
import numpy as np
from tensorflow import keras

# Keras: Framework in TensorFlow for making the definition of training and using ML models easier.
# Units: 1 Layer
# Input Shape: 1
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Optimizer Function: Stochastic Gradient Descent (SGD)
# Loss Function: Mean Squared Error
model.compile(optimizer='sgd', loss='mean_squared_error')

# Relationship: y = 3x + 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Epochs: Amount of times to run the model for training
model.fit(xs, ys, epochs=500)

# Guess the result of the training.
print(model.predict([10.0]))

