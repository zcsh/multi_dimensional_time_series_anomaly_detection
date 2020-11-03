from tensorflow.keras import layers
import tensorflow as tf


class LSTMEncoder(layers.Layer):
    def __init__(self, feature_config, encoder_config):
        super(LSTMEncoder, self).__init__()
        self.feature_config = feature_config
        self.encoder_config = encoder_config
        self.layers_info = self.encoder_config['layers']

        self.embedding_layers = {}
        for i, feature_info in enumerate(self.feature_config):
            if feature_info.get('embedding_size', -1) > 0:
                self.embedding_layers[feature_info['name']] = layers.Embedding(feature_info['n_values'],
                                                                               feature_info['embedding_size'])

        self.lstm_layers = [layers.LSTM(units=layer_info[0],
                                        activation=layer_info[1],
                                        return_sequences=True)
                            for layer_info in self.layers_info[:-1]]
        self.lstm_layers.append(
            layers.LSTM(units=self.layers_info[-1][0],
                        activation=self.layers_info[-1][1],
                        return_sequences=False))

    def call(self, x):
        # input shape(batch_size, series_len, dimension)
        inputs = []
        for i, feature_info in enumerate(self.feature_config):
            if feature_info.get('embedding_size', -1) > 0:
                embedded = self.embedding_layers[feature_info['name']](x[:, :, i])
                inputs.append(embedded)
            else:
                inputs.append(x[:, :, i:i + 1])

        x = layers.concatenate(inputs, axis=-1)

        for layer in self.lstm_layers:
            x = layer(x)

        return x


class ConvEncoder(layers.Layer):
    def __init__(self, feature_config, encoder_config):
        super(ConvEncoder, self).__init__()
        self.feature_config = feature_config
        self.encoder_config = encoder_config
        self.layers_info = self.encoder_config['layers']

        self.embedding_layers = {}
        for i, feature_info in enumerate(self.feature_config):
            if feature_info.get('embedding_size', -1) > 0:
                self.embedding_layers[feature_info['name']] = layers.Embedding(feature_info['n_values'],
                                                                               feature_info['embedding_size'])

        self.conv_layers = [layers.Conv1D(filters=layer_info[0],
                                          kernel_size=layer_info[1],
                                          strides=1,
                                          activation=layer_info[2])
                            for layer_info in self.layers_info[:]]
        self.pooling_layers = [layers.AvgPool1D(pool_size=2) for _ in range(len(self.conv_layers))]

    def call(self, x):
        # input shape(batch_size, series_len, dimension)
        inputs = []
        for i, feature_info in enumerate(self.feature_config):
            if feature_info.get('embedding_size', -1) > 0:
                embedded = self.embedding_layers[feature_info['name']](x[:, :, i])
                inputs.append(embedded)
            else:
                inputs.append(x[:, :, i:i + 1])

        x = layers.concatenate(inputs, axis=-1)

        for conv_layer, pooling_layer in zip(self.conv_layers, self.pooling_layers):
            x = conv_layer(x)
            x = pooling_layer(x)

        x = tf.squeeze(x, axis=1)

        return x
