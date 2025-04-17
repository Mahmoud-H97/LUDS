import numpy as np
import matplotlib.pyplot as plt
from keras.api._v2.keras import activations
import keras
import keras.utils
from sklearn.metrics import mean_squared_error

# Model Structure
mod = keras.Sequential(
    [
        keras.layers.Dense(units=5, input_shape=[1], activation='ReLU'),
        keras.layers.Dense(units=4, activation='tanh'),
        keras.layers.Dense(units=4, activation='tanh'),
        keras.layers.Dense(units=1, activation='softmax')
    ]
)

keras.utils.plot_model(mod,show_shapes=True, show_layer_names=True, show_layer_activations=True)
