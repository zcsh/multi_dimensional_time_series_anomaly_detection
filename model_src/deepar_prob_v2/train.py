from __future__ import absolute_import
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, losses
from .config import config
from .model import DeepAR
from .loss import MeanNaiveError


def train(device_name, data):
    x, y = data

    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1"],
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    with mirrored_strategy.scope():
        # loss_type = config.get('model_architecture').get('loss', 'NegLogLikelihood')
        # loss = getattr(__import__('loss'), loss_type)()
        loss = MeanNaiveError()

        # metric_list = config.get('model_architecture').get('metrics', [])
        # metric_list = [getattr(__import__('metric'), metric)() for metric in metric_list]

        optimizer = optimizers.Adam()

        model = DeepAR(config)
        model.compile(optimizer=optimizer, loss=loss)  # , metrics=metric_list)

        early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=1e-6,
                                                          patience=5,
                                                          restore_best_weights=True,
                                                          verbose=1)

        ckpt_path = config.get('ckpt_path', './')
        ckpt_path = os.path.join(ckpt_path, device_name)
        model_checkpoint_callback = callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                              monitor='val_loss',
                                                              save_best_only=True,
                                                              save_weight_only=True)

        callback_list = [early_stopping_callback, model_checkpoint_callback]

    history = model.fit(x, y, callbacks=callback_list, batch_size=32, epochs=50, validation_split=0.2)

    history.model.save(os.path.join(config['save_path'], device_name), save_format='tf')

    return history


def test():
    device_name = 'test'

    n_target = 0
    features_info = config['features']
    for feature_info in features_info:
        if feature_info.get('type') == 'target':
            n_target += 1

    n_sample = 1000
    series_len = 5

    inputs = []
    for feature_info in features_info:
        if feature_info.get('embedding_size', -1) > 0:
            inputs.append(np.random.randint(feature_info.get('embedding_size'), size=(n_sample, series_len, 1)))
        else:
            inputs.append(np.random.random(size=(n_sample, series_len, 1)))
    x = np.concatenate(inputs, axis=-1)
    y = np.zeros(shape=(n_sample,))
    print(x.shape, y.shape)

    history = train(device_name, (x, y))
    return history


if __name__ == '__main__':
    test()
