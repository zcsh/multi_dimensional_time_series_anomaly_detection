import tensorflow as tf
from tensorflow.keras.layers import Layer


class NegLogLikelihood(Layer):
    def __init__(self):
        super(NegLogLikelihood, self).__init__()

    def call(self, truth, pred):
        miu, cov_inv = pred[:, 0, :], pred[:, 1:, :]

        truth = tf.expand_dims(truth, 1)  # shape(batch_size, 1, dimension)
        miu = tf.expand_dims(miu, 1)  # shape (batch_size, 1, dimension)

        diff = truth - miu  # shape(batch_size, 1, dimension)
        diffT = tf.transpose(diff, [0, 2, 1])  # shape(batch_size, dimension, 1)
        neg_log_likelihood = 0.5 * tf.squeeze(
            tf.linalg.matmul(tf.linalg.matmul(diff, cov_inv), diffT), axis=[1, 2]) - tf.math.log(
            tf.math.sqrt(tf.linalg.det(cov_inv)))  # shape (batch_size,)
        # neg_log_likelihood = tf.math.reduce_mean(neg_log_likelihood)

        return neg_log_likelihood
