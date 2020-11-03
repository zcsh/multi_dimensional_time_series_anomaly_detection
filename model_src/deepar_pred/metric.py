from tensorflow.keras.metrics import Metric
import tensorflow as tf


class MeanErrorMetric(Metric):
    def __init__(self, name):
        super(MeanErrorMetric, self).__init__(name)
        self.total_error = self.add_weight(name='total_error', dtype=tf.float32, initializer='zeros')
        self.n = self.add_weight(name='n', dtype=tf.int32, initializer='zeros')

    def _calc_error_sum(self, truth, miu):
        raise NotImplementedError

    def update_state(self, truth, pred):
        miu, _ = pred[:, 0, :], pred[:, 1:, :]
        self.total_error.assign_add(self._calc_error_sum(truth, miu))
        self.n.assign_add(tf.shape(truth)[0])

    def result(self):
        return self.total_error / tf.cast(self.n, tf.float32)


class MSE(MeanErrorMetric):
    def __init__(self, name='mse'):
        super(MSE, self).__init__(name)

    def _calc_error_sum(self, truth, miu):
        return tf.reduce_sum(tf.math.square(truth - miu))


class MAE(MeanErrorMetric):
    def __init__(self, name='mae'):
        super(MAE, self).__init__(name)

    def _calc_error_sum(self, truth, miu):
        return tf.reduce_sum(tf.math.abs(truth - miu))


class MAPE(MeanErrorMetric):
    def __init__(self, name='mape'):
        super(MAPE, self).__init__(name)

    def _calc_error_sum(self, truth, miu):
        return tf.reduce_sum(tf.math.abs((truth - miu) / (miu + tf.keras.backend.epsilon())))

# class MSE(Metric):
#     def __init__(self, name="mse"):
#         super(MSE, self).__init__(name)
#         self.total_se = self.add_weight(name='total_se', dtype=tf.float32, initializer='zeros')
#         self.n = self.add_weight(name='n', dtype=tf.int32, initializer='zeros')
#
#     def update_state(self, truth, pred):
#         miu, cov_inv = pred
#         self.total_se.assign_add(tf.reduce_sum(tf.math.square(truth - miu)))
#         self.n.assign_add(tf.shape(truth)[0])
#
#     def result(self):
#         return self.total_se / tf.cast(self.n, tf.float32)
#
#
# class MAE(Metric):
#     def __init__(self, name="mae"):
#         super(MAE, self).__init__(name)
#         self.total_ae = self.add_weight(name='total_se', dtype=tf.float32, initializer='zeros')
#         self.n = self.add_weight(name='n', dtype=tf.int32, initializer='zeros')
#
#     def update_state(self, truth, pred):
#         miu, cov_inv = pred
#         self.total_ae.assign_add(tf.reduce_sum(tf.math.abs(truth - miu)))
#         self.n.assign_add(tf.shape(truth)[0])
#
#     def result(self):
#         return self.total_ae / tf.cast(self.n, tf.float32)
