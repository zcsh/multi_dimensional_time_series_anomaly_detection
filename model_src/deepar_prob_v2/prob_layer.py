import tensorflow as tf
from tensorflow.keras.layers import Layer


class NegLogLikelihood(Layer):
    def __init__(self):
        super(NegLogLikelihood, self).__init__()

    def call(self, truth, pred):
        miu, upper = pred[:, 0, :], pred[:, 1:, :]

        upperT = tf.transpose(upper, [0, 2, 1])
        cov_inv = tf.matmul(upper, upperT)

        upper_det = tf.reduce_prod(tf.linalg.diag_part(upper), axis=-1)
        cov_inv_det = tf.math.square(upper_det)

        truth = tf.expand_dims(truth, 1)  # shape(batch_size, 1, dimension)
        miu = tf.expand_dims(miu, 1)  # shape (batch_size, 1, dimension)

        diff = truth - miu  # shape(batch_size, 1, dimension)
        diffT = tf.transpose(diff, [0, 2, 1])  # shape(batch_size, dimension, 1)
        neg_log_likelihood = 0.5 * tf.squeeze(
            tf.linalg.matmul(tf.linalg.matmul(diff, cov_inv), diffT), axis=[1, 2]) - tf.math.log(
            tf.math.sqrt(cov_inv_det))  # shape (batch_size,)

        return neg_log_likelihood
