import tensorflow as tf
import numpy as np

from model.utilities import downscale, pad
from model.featureExtractor import FeatureExtractor
from model.coordExtractor import CoordExtractor

class GlobalStage(tf.keras.layers.Layer):
    def __init__(self):
        super(GlobalStage, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.x_coord_extractor = CoordExtractor(640)
        self.y_coord_extractor = CoordExtractor(256)

    def call(self, input):
        #input = downscale(input, (128, 320))

        print(f"Shape: {input.shape}")
        input = self.feature_extractor(input)
        x1 = self.x_coord_extractor(input)
        y1 = self.y_coord_extractor(input)

        return x1, y1


class LocalStage(tf.keras.layers.Layer):
    def __init__(self):
        super(LocalStage, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.x_coord_extractor = CoordExtractor(640)
        self.y_coord_extractor = CoordExtractor(256)

    def call(self, input):
        # print("Begin crop")
        # input = crop(input, (x1, y1), (128, 320))
        # print("Finished cropping")
        input = self.feature_extractor(input)
        x2 = self.x_coord_extractor(input)
        y2 = self.y_coord_extractor(input)

        return x2, y2 # Check this dimension

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.global_stage = GlobalStage()
        self.local_stage = LocalStage()

    def call(self, input):
        #print(f"beginning shape: {input.shape}")
        _, _, h0, w0, _ = input.shape
        x1, y1 = self.global_stage(input)

        #print(f"x1 shape: {x1.shape}, y1 shape: {y1.shape}")
        """
        _, h1 = y1.shape
        _, w1 = x1.shape

        x1_guess = tf.argmax(x1, axis=1)
        y1_guess = tf.argmax(y1, axis=1)

        x1_guess = tf.cast(x1_guess, tf.float32)
        y1_guess = tf.cast(y1_guess, tf.float32)
        w0 = tf.cast(w0, tf.float32)
        w1 = tf.cast(w1, tf.float32)
        h0 = tf.cast(h0, tf.float32)
        h1 = tf.cast(h1, tf.float32)

        x1_guess = tf.cast(x1_guess*w0/w1, tf.int32)
        y1_guess = tf.cast(y1_guess*h0/h1, tf.int32)
        #print(f"x1_guess shape: {x1_guess.shape}, y1_guess shape: {y1_guess.shape}")
        x2, y2 = self.local_stage(input, x1_guess, y1_guess)
        #print(f"x2 shape: {x2.shape}, y2 shape: {y2.shape}")

        w0 = tf.cast(w0, tf.int32)
        w1 = tf.cast(w1, tf.int32)
        h0 = tf.cast(h0, tf.int32)
        h1 = tf.cast(h1, tf.int32)

        x = pad(x2, w0, w1, x1_guess)
        y = pad(y2, h0, h1, y1_guess)

        #print(f"h0: {h0}, w0: {w0}, h1: {h1}, w1: {w1}")

        #print(f"Final x, y shape: {x.shape}, {y.shape}")

        return x, y
        """

class GlobalModel(tf.keras.Model):
    def __init__(self):
        super(GlobalModel, self).__init__()

        self.global_stage = GlobalStage()

    def call(self, input):
        #print(f"beginning shape: {input.shape}")
        x, y = self.global_stage(input)

        return x, y

class LocalModel(tf.keras.Model):
    def __init__(self):
        super(LocalModel, self).__init__()

        self.local_stage = LocalStage()

    def call(self, input):
        print(f"beginning shape: {input.shape}")
        x, y = self.local_stage(input)

        #print(f"h0: {h0}, w0: {w0}, h1: {h1}, w1: {w1}")
        return x, y
