from tensorflow.keras.losses import Loss
import tensorflow as tf


class MeanNaiveError(Loss):
    def __init__(self):
        super(MeanNaiveError, self).__init__()

    def call(self, truth, pred):
        return tf.math.reduce_mean(pred - truth)
