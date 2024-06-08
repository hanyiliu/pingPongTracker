import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.activations import sigmoid

class CoordExtractor(tf.keras.layers.Layer):
    def __init__(self, units, rate=0.3):
        super(CoordExtractor, self).__init__()
        self.fullyC1 = Dense(units)
        self.relu = ReLU()
        self.dropout = Dropout(rate)
        self.fullyC2 = Dense(units // 2)


    def call(self, x):
        x = self.fullyC1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fullyC2(x)
        x = sigmoid(x)

        return x
