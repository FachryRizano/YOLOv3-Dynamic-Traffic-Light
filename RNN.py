import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
model = Sequential()

model = Sequential()

model.add(SimpleRNN(units=32, input_shape=(4,1), activation="relu"))
model.add(Dense(8, activation="relu")) 
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
# model.summary()