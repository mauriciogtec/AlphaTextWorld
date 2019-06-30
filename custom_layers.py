# This code contains modified code from
# https://github.com/Lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations,\
    optimizers, utils, layers, models, regularizers, initializers


class LayerNormalization(layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1],
            initializer='ones',
            trainable=True)
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class IdentityBlock(models.Model):
    """A residual block part of the encoder, similar to ResNet and AlphaZero"""
    def __init__(self, filters, kernel_size, l2=None, **kwargs):
        super(IdentityBlock, self).__init__(**kwargs)
        self.layer_norm_1 = LayerNormalization()
        self.relu_1 = layers.Activation('relu')
        self.conv_1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(l2))
        self.layer_norm_2 = layers.LayerNormalization()
        self.relu_2 = layers.Activation('relu')
        self.conv_2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(l2))
        self.add_layer = layers.Add()

    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        x = self.relu_1(x)
        x = self.conv_1(x)
        x = self.layer_norm_2(x)
        x = self.relu_2(x)
        x = self.conv_2(x)
        x = self.add_layer([x, inputs])
        return x


class LocalFeaturesExtractor(models.Model):
    """Transforms text to memory and output vectors"""
    def __init__(self, filters, kernel_size,
                 num_blocks, l2=None, **kwargs):
        super(LocalFeaturesExtractor, self).__init__(**kwargs)
        self.layer_norm_1 = LayerNormalization()
        self.relu_1 = layers.Activation('relu')
        self.conv_1 = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding='same',
            kernel_regularizer=regularizers.l2(l2))
        self.blocks = [
            IdentityBlock(
                filters=filters,
                kernel_size=3,
                l2=l2)
            for i in range(num_blocks)]

    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        x = self.relu_1(x)
        x = self.conv_1(x)
        for block in self.blocks:
            x = block(x)
        return x


class ScaledDotProductAttention(models.Model):
    def __init__(self, dropout=0.1, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        # self.dropout = layers.Dropout(dropout)

    def call(self, queries, keys, values,
             training=False, attentions=False):
        """
        inputs
            queries: (queries_size x pool_dim)
            keys: (kbatch x keys_size x pool_dim)
            values: (kbatch x keys_size x output_dim)
        returns
            output: (kbatch x queries_size x output_dim)
        """
        attn = tf.matmul(queries, keys, transpose_b=True)  # kb x qs x ks
        attn /= np.sqrt(keys.shape[-1])
        attn = tf.math.softmax(attn, axis=-1)  # kbatch x qsize x ksize
        # if training:
        #     attn = self.dropout(attn)
        output = tf.matmul(attn, values)  # kbatch x qsize x odim

        return output


class MultiHeadAttention(models.Model):
    def __init__(self, units, num_heads, residual=True,
                 l2=None, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.units_per_head = units // num_heads
        self.residual = residual

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        for _ in range(num_heads):
            self.qs_layers.append(
                layers.Dense(
                    self.units_per_head,
                    use_bias=False,
                    kernel_regularizer=regularizers.l2(l2)))
            self.ks_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        self.units_per_head,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(l2))))
            self.vs_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        use_bias=False,
                        units=self.units_per_head,
                        kernel_regularizer=regularizers.l2(l2))))

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dense = layers.TimeDistributed(
            layers.Dense(
                units,
                use_bias=True,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2)))
        # self.dropout = layers.Dropout(dropout)
        self.layernorm = LayerNormalization()

    def call(self, queries, keys, training=False):
        """
        inputs
            queries: (queries_size x queries_dim)
            keys: (keys_batch x keys_size x keys_dim)
        returns
            output: (keys_batch x queries_size x output_dim)
        """
        heads = []

        for i in range(self.num_heads):
            Q = self.qs_layers[i](queries)  # qsize x hdim
            K = self.ks_layers[i](keys)  # kbatch x ksize x hdim
            V = self.vs_layers[i](keys)  # kbatch x ksize x hdim
            head = self.attention(Q, K, V, training=training)  # kb x qs x hd
            heads.append(head)

        output = tf.concat(heads, axis=2)  # kbatch x qsize x units
        output = self.dense(output)  # kbatch x qsize x units
        # if training:
        #     output = self.dropout(output)
        if self.residual:
            output += queries
        output = self.layernorm(output)
        return output


class AttentionEncoder(models.Model):
    """Transforms text to memory and output vectors"""
    def __init__(self, units, num_heads, num_blocks,
                 residual=True, l2=None, **kwargs):
        #
        super(AttentionEncoder, self).__init__(**kwargs)
        self.residual = residual
        #  note: units here shouldnt be free parameter, its always like input
        self.blocks = [
            MultiHeadAttention(
                units=units,
                num_heads=num_heads,
                residual=self.residual,
                l2=l2)
            for i in range(num_blocks)]

    def call(self, queries, keys, training=False):
        """
        inputs
            queries: (queries_size x queries_dim)
            keys: (keys_batch x keys_size x keys_dim)
        returns
            output: (keys_batch x queries_size x queries_dim)
        """
        x = queries
        for block in self.blocks:
            x = block(x, keys, training=training)
        return x


class PairedScaledDotProductAttention(models.Model):
    def __init__(self, dropout=0.1, **kwargs):
        super(PairedScaledDotProductAttention, self).__init__(**kwargs)
        # self.dropout = layers.Dropout(dropout)

    def call(self, queries, keys, values,
             training=False, attentions=False):
        """
        inputs
            queries: (queries_size x pool_dim)
            keys: (queries_size x keys_size x pool_dim)
            values: (queries_size x keys_size x output_dim)
        returns
            output: (queries_size x output_dim)
        """
        queries = tf.expand_dims(queries, 1)  # qsize x 1 x pdim
        attn = tf.matmul(queries, keys, transpose_b=True)  # qs x 1 x ks
        attn /= np.sqrt(keys.shape[-1])
        attn = tf.math.softmax(attn, axis=-1)  # qsize x ksize
        # if training:
        #     attn = self.dropout(attn)
        output = tf.matmul(attn, values)  # qsize x 1 x odim
        output = tf.squeeze(output, axis=1)

        return output


class PairedMultiHeadAttention(models.Model):
    def __init__(self, units, num_heads, residual=True,
                 l2=None, dropout=0.1, **kwargs):
        super(PairedMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.units_per_head = units // num_heads
        self.residual = residual

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        for _ in range(num_heads):
            self.qs_layers.append(
                layers.Dense(
                    self.units_per_head,
                    use_bias=False,
                    kernel_regularizer=regularizers.l2(l2)))
            self.ks_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        self.units_per_head,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(l2))))
            self.vs_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        use_bias=True,
                        units=self.units_per_head,
                        kernel_regularizer=regularizers.l2(l2))))

        self.attention = PairedScaledDotProductAttention(dropout=dropout)
        self.dense = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2))
        # self.dropout = layers.Dropout(dropout)
        self.layernorm = LayerNormalization()

    def call(self, queries, keys, training=False):
        """
        inputs
            queries: (queries_size x queries_dim)
            keys: (queries_size x keys_size x keys_dim)
        returns
            output: (queries_size x queries_dim)
        """
        heads = []

        for i in range(self.num_heads):
            Q = self.qs_layers[i](queries)  # qsize x hdim
            K = self.ks_layers[i](keys)  # qsize x ksize x hdim
            V = self.vs_layers[i](keys)  # qsize x ksize x hdim
            head = self.attention(Q, K, V, training=training)  # qs x hd
            heads.append(head)

        output = tf.concat(heads, axis=-1)  # qsize x units
        output = self.dense(output)  # qsize x units
        # if training:
        #     output = self.dropout(output)
        if self.residual:
            output += queries
        output = self.layernorm(output)
        return output


class PairedAttentionEncoder(models.Model):
    """Transforms text to memory and output vectors"""
    def __init__(self, units, num_heads, num_blocks,
                 residual=True, l2=None, **kwargs):
        #
        super(PairedAttentionEncoder, self).__init__(**kwargs)
        self.residual = residual
        #  note: units here shouldnt be free parameter, its always like input
        self.blocks = [
            PairedMultiHeadAttention(
                units=units,
                num_heads=num_heads,
                residual=self.residual,
                l2=l2)
            for i in range(num_blocks)]

    def call(self, queries, keys, training=False):
        """
        inputs
            queries: (queries_size x queries_dim)
            keys: (queries_size x keys_size x keys_dim)
        returns
            output: (queries_size x queries_dim)
        """
        x = queries
        for block in self.blocks:
            x = block(x, keys, training=training)
        return x


class DenseHead(models.Model):
    """Trained to predict a singlue value"""
    def __init__(self, hidden_units,
                 residual=True, dropout=0.25, l2=None, **kwargs):
        super(DenseHead, self).__init__(**kwargs)
        self.dense_1 = layers.Dense(
            units=hidden_units,
            kernel_regularizer=regularizers.l2(l2),
            activation='relu')
        # self.dropout = layers.Dropout(dropout)
        self.dense_2 = layers.Dense(
            units=1,
            kernel_regularizer=regularizers.l2(l2))
        self.residual = residual

    def call(self, inputs, training=None):
        """
        inputs
            inputs: (inputs_size x input_dim)
        returns
            output: (inputs_size)
        """
        x = inputs
        x = self.dense_1(x)  # isize x hdim
        # if training:
        #     x = self.dropout(x)
        x = self.dense_2(x)  # isize x 1
        x = tf.squeeze(x, axis=1)  # isize
        return x

