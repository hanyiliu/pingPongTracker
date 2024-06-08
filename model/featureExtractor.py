import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.activations import sigmoid

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.conv = Conv2D(filters, (3, 3), padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, rate=0.3):
        super(FeatureExtractor, self).__init__()
        self.conv1 = Conv2D(64, (1, 1))
        self.bn = BatchNormalization()
        self.relu1 = ReLU()

        self.convlayer1 = ConvBlock(filters=64)
        self.convlayer2 = ConvBlock(filters=64)
        self.dropout1 = Dropout(rate)

        self.convlayer3 = ConvBlock(filters=128)
        self.convlayer4 = ConvBlock(filters=128)
        self.dropout2 = Dropout(rate)

        self.convlayer5 = ConvBlock(filters=256)
        self.convlayer6 = ConvBlock(filters=256)
        self.dropout3 = Dropout(rate)

        self.flatten = Flatten()
        self.fullyC = Dense(1792)
        self.relu2 = ReLU()
        self.dropout4 = Dropout(rate)


    def call(self, x):
        # print("Passing through conv1")
        x = self.conv1(x)
        #print(f"Finished passing through conv1: {x.shape}")
        x = self.bn(x)
        x = self.relu1(x)
        #print(f"Check 1: {x.shape}")
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.dropout1(x)
        #print(f"Check 2: {x.shape}")
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.dropout2(x)
        #print(f"Check 3: {x.shape}")
        x = self.convlayer5(x)
        x = self.convlayer6(x)
        x = self.dropout3(x)
        #print(f"Check 4: {x.shape}")
        x = self.flatten(x)
        x = self.fullyC(x)
        x = self.relu2(x)
        x = self.dropout4(x)
        #print(f"Check 5: {x.shape}")
        return x
