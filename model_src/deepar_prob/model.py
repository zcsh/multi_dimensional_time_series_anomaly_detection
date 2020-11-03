import tensorflow as tf
from tensorflow.keras import Model


class DeepAR(Model):
    """
    the model is designed to:
    1. predict the expectation and covariance of current target states,
        given current covariates and past targets and covariates.
    2. output the likelihood of the ground truth of the current states,
        given the predicted expectation and covariance.
    """

    def __init__(self, config):
        super(DeepAR, self).__init__()
        self.feature_config = config['features']
        self.encoder_config = config['model_architecture']['encoder']
        self.decoder_config = config['model_architecture']['decoder']
        self.target_indices = [i for i in range(len(config['features']))
                               if config['features'][i]['type'] == 'target']
        self.covariate_indices = [i for i in range(len(config['features']))
                                  if config['features'][i]['type'] == 'covariate']

        self.encoder = getattr(__import__('encoder'), self.encoder_config['type'])(self.feature_config,
                                                                                   self.encoder_config)
        self.decoder = getattr(__import__('decoder'), self.decoder_config['type'])(self.decoder_config)
        self.prob_layer = getattr(__import__('prob_layer'), config['model_architecture']['prob_layer'])()

    def call(self, x):
        targets = []
        for i in self.target_indices:
            targets.append(x[:, :-1, i:i + 1])
        targets = tf.concat(targets, axis=-1)

        covariates = []
        for i in self.covariate_indices:
            covariates.append(x[:, 1:, i:i + 1])
        covariates = tf.concat(covariates, axis=-1)

        truth = []
        for i in self.target_indices:
            truth.append(x[:, -1, i:i + 1])
        truth = tf.concat(truth, axis=-1)

        inputs = tf.concat([targets, covariates], axis=-1)
        encoded = self.encoder(inputs)
        miu, cov_inv = self.decoder(encoded)
        predicted = tf.concat([tf.expand_dims(miu, axis=1), cov_inv], axis=1)
        likelihood = self.prob_layer(truth, predicted)

        return likelihood
