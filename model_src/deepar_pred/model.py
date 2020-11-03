import tensorflow as tf
from tensorflow.keras import Model


class DeepAR(Model):
    def __init__(self, config):
        super(DeepAR, self).__init__()
        self.feature_config = config['features']
        self.encoder_config = config['model_architecture']['encoder']
        self.decoder_config = config['model_architecture']['decoder']

        self.encoder = getattr(__import__('encoder'), self.encoder_config['type'])(self.feature_config,
                                                                                   self.encoder_config)
        self.decoder = getattr(__import__('decoder'), self.decoder_config['type'])(self.decoder_config)

    def call(self, x):
        encoded = self.encoder(x)
        miu, cov_inv = self.decoder(encoded)
        return tf.concat([tf.expand_dims(miu, axis=1), cov_inv], axis=1)
