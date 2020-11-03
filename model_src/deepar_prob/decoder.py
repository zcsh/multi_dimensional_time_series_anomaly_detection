import tensorflow as tf
from tensorflow.keras import layers


# import tensorflow_probability as tfp


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


class DecoderFull(layers.Layer):
    def __init__(self, config):
        super(DecoderFull, self).__init__()
        self.config = config
        self.n_target = self.config['n_target']

        self.miu_net = layers.Dense(self.n_target, activation=None)
        self.cov_net = layers.Dense(self.n_target, activation=None)

    def call(self, x):
        miu = self.miu_net(x)

        x = tf.reshape(x, (-1, self.n_target, tf.cast(self.config['input_dim'] / self.config['n_target'], tf.int32)))
        full = self.cov_net(x)

        upper = tf.linalg.band_part(full, 0, -1)
        diag = tf.linalg.diag_part(full)
        diag = tf.linalg.diag(diag)
        upper -= diag
        diag = tf.math.square(diag) + tf.eye(self.n_target) * 1e-2
        upper += diag
        upperT = tf.transpose(upper, [0, 2, 1])

        cholesky = tf.matmul(upper, upperT)

        return miu, cholesky

# class DecoderFullTFP(layers.Layer):
#     def __init__(self, config):
#         super(DecoderFullTFP, self).__init__()
#         self.config = config
#         self.n_target = self.config['n_target']
#
#         self.miu_net = layers.Dense(self.n_target, activation=None)
#         self.cov_net = layers.Dense(tf.cast((self.n_target + 1) / 2 * self.n_target, tf.int32), activation=None)
#
#     def call(self, x):
#         miu = self.miu_net(x)
#         flattened = self.cov_net(x)
#         upper = tfp.math.fill_triangular(flattened)
#         diag = tf.linalg.diag_part(upper)
#         diag = tf.linalg.diag(diag)
#         upper -= diag
#         diag = tf.math.square(diag) + 1e-2
#         upper += diag
#         upperT = tf.transpose(upper, [0, 2, 1])
#
#         cholesky = tf.matmul(upper, upperT)
#
#         return miu, cholesky
