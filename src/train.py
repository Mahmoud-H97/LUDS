import keras
import keras.utils
from mlp import mod
# from cnn import mod


# build model
mod.build()
print(mod.summary())

# Model compilation
mod.compile(optimizer=keras.optimizer.Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['accuracy'])

model.fit
