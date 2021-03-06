import tensorflow as tf
from tensorflow.keras import layers


class DecoderDiag(layers.Layer):
    def __init__(self, config):
        super(DecoderDiag, self).__init__()
        self.config = config
        self.n_target = self.config['n_target']

        self.miu_net = layers.Dense(self.n_target, activation=None)
        self.cov_net = layers.Dense(self.n_target, activation=None)

    def call(self, x):
        miu = self.miu_net(x)
        cov_inv = self.cov_net(x)
        # cov_inv = tf.math.log(1. + tf.math.exp(cov_inv))
        cov_inv = tf.math.square(cov_inv) + 1e-5
        cov_inv = tf.linalg.diag(cov_inv)
        return miu, cov_inv

# class DecoderFull(layers.Layer):
#     def __init__(self, config):
#         super(DecoderFull, self).__init__()
#         self.config = config
#         self.n_target = self.config['n_target']
#
#         self.miu_net = layers.Dense(self.n_target, activation=None)
#         self.cov_net = layers.Dense(self.n_target, activation=None)
#
#     def call(self, x):
#         miu = self.miu_net(x)
#
#         x = tf.reshape(x, (-1, self.n_target, tf.cast(self.config['input_dim'] / self.config['n_target'], tf.int32)))
#         cov_inv = self.cov_net(x)
#         upper = tf.linalg.band_part(cov_inv, 0, -1)
#         diag = tf.linalg.diag_part(cov_inv)
#         diag = tf.linalg.diag(diag)
#         upper -= diag
#         lower = tf.transpose(upper, [0, 2, 1])
#         diag = tf.math.log(1. + tf.math.exp(diag))
#         cov_inv = upper + lower + diag
#         return miu, diag
