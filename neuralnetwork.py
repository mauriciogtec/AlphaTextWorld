import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, initializers, \
    regularizers, optimizers, utils, layers, models
import textutils
import pdb


class ResidualBlock(layers.Layer):
    """A residual block part of the encoder, similar to ResNet and AlphaZero"""
    def __init__(self, filters, kernel_size, l2=None, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.batch_norm_1 = layers.BatchNormalization()
        self.relu_1 = layers.Activation('relu')
        self.conv_1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(l2))
        self.batch_norm_2 = layers.BatchNormalization()
        self.relu_2 = layers.Activation('relu')
        self.conv_2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(l2))
        self.add_layer = layers.Add()

    def call(self, inputs, training=None):
        x = self.batch_norm_1(inputs, training=training)
        x = self.relu_1(x)
        x = self.conv_1(x)
        x = self.batch_norm_2(x, training=training)
        x = self.relu_2(x)
        x = self.conv_2(x)
        x = self.add_layer([x, inputs])
        return x


class BahdanauAttention(layers.Layer):
    """
    Implements a module for BahdanauAttention
    Taken from www.tensorflow.org/alpha/tutorials/text/nmt_with_attention
    """
    def __init__(self, units, l2=None, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.W1 = layers.Dense(
            units=units,
            kernel_regularizer=regularizers.l2(l2))
        self.W2 = layers.Dense(
            units=units,
            kernel_regularizer=regularizers.l2(l2))
        self.V = layers.Dense(
            units=1,
            kernel_regularizer=regularizers.l2(l2))

    def call(self, query, values):
        # we are doing this to perform addition to calculate the score
        hidden_time_axis = tf.expand_dims(query, 1)
        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.math.tanh(self.W1(values) + self.W2(hidden_time_axis)))
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.math.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.math.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class StateEncoder(layers.Layer):
    """Transforms text to memory and output vectors"""
    def __init__(self, filters, kernel_size,
                 num_residual_blocks, l2=None, **kwargs):
        super(StateEncoder, self).__init__(**kwargs)
        self.residual_blocks = [
            ResidualBlock(
                filters=filters,
                kernel_size=3,
                l2=l2,
                name="residual_block_{:d}".format(i))
            for i in range(num_residual_blocks)]
        self.recurrency = layers.Bidirectional(
            layers.GRU(
                units=filters,
                stateful=False,
                return_sequences=True,
                return_state=True,
                implementation=2,  # GPU compatible
                kernel_regularizer=regularizers.l2(l2)),
            name="recurrency")
        self.concatenate = layers.Concatenate()
        self.attention = BahdanauAttention(filters)

    def call(self, inputs, training=None):
        x = inputs
        for block in self.residual_blocks:
            x = block(x, training=training)
        x, fwd, bwd = self.recurrency(x)
        query = layers.concatenate([fwd, bwd])
        x, _ = self.attention(query, x)
        return x

class DenseHead(layers.Layer):
    """Trained to predict a singlue value"""
    def __init__(self, units, dropout=0.5, l2=None, **kwargs):
        super(DenseHead, self).__init__(**kwargs)
        self.dense_1 = layers.Dense(
            units=units,
            kernel_regularizer=regularizers.l2(l2),
            name="dense_1",
            activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.dense_2 = layers.Dense(
            units=1,
            kernel_regularizer=regularizers.l2(l2),
            name="dense_2")
    
    def call(self, inputs, training=None):
        x = inputs
        x = self.dense_1(x)
        if training:
            x = self.dropout(x)
        x = self.dense_2(x)
        return x


class AlphaTextWorldNet(models.Model):
    """
    Learn to play from memory
    """
    REG_PENALTY = 1e-3

    def __init__(self,  embeddings, vocab, **kwargs):
        super(AlphaTextWorldNet, self).__init__(**kwargs)
        
        self.memory = []
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for i, w in self.word2id.items()}
        
        embedding_dim, vocab_size = embeddings.shape
        self.embeddings = layers.Embedding(
            input_dim=vocab_size,
            input_length=None,
            output_dim=embedding_dim,
            embeddings_initializer=initializers.Constant(embeddings),
            trainable=False,
            name="embeddings")

        self.obs_encoder = StateEncoder(
            filters=embedding_dim,
            kernel_size=3,
            num_residual_blocks=8,
            l2=self.REG_PENALTY,
            name="obs_encoder")

        self.cmd_encoder = StateEncoder(
            filters=embedding_dim,
            kernel_size=2,
            num_residual_blocks=2,
            l2=self.REG_PENALTY,
            name="cmd_encoder")

        self.memory_attention = BahdanauAttention(
            units=embedding_dim,
            l2=self.REG_PENALTY,
            name="memory_attention")

        self.value_head = DenseHead(
            units=128,
            dropout=0.5,
            l2=self.REG_PENALTY,
            name="value_head")
        
        self.policy_head = DenseHead(
            units=128,
            dropout=0.5,
            l2=self.REG_PENALTY,
            name="policy_head")
    
    def encode_text(self, textlist, encoder="obs", training=None):
        """common pattern: embed -> encode"""
        assert encoder in ["obs", "cmd"]
        x = textutils.text2tensor(textlist, self.word2id)
        x = self.embeddings(x)
        if encoder == "obs":
            x = self.obs_encoder(x, training=training)
        else:
            x = self.cmd_encoder(x, training=training)
        return x

    def call(self, obs, cmdlist, recall=True, training=None):
        # fully process observation with attention
        obsx = self.encode_text([obs], encoder="obs", training=training)
        if len(self.memory) > 0:
            memoryx = self.encode_text(self.memory, encoder="obs", training=training)
            memoryx = tf.expand_dims(memoryx, axis=0)  # (1 x mem_size x hidden_size)
            obsxplus, _ = self.memory_attention(obsx, memoryx)
            obsx += obsxplus
        else:
            memoryx = tf.zeros((1, 1, obsx.shape[1]))

        # fully process command with attention
        cmdlistx = self.encode_text(cmdlist, encoder="cmd", training=training)
        memoryx = tf.concat([memoryx, tf.expand_dims(obsx, axis=0)], axis=1)
        memoryx = tf.stack([tf.squeeze(memoryx, axis=0)] * len(cmdlist), axis=0)
        cmdlistxplus, _ = self.memory_attention(cmdlistx, memoryx)
        cmdlistx += cmdlistxplus # cmd_size x hidden

        # value heead from obs memory only
        value = self.value_head(obsx)
        value = tf.squeeze(value)
        
        # policy head from obs memory and commands
        x = tf.stack([tf.squeeze(obsx, axis=0)] * len(cmdlist), axis=0)
        x = tf.concat([x, cmdlistx], axis=1)  # (cmd_size x (hidden_obs + hidden_cmds))
        policy_logits = self.policy_head(x)  # cmd_size x 1
        policy = tf.math.softmax(policy_logits, axis=0) # cmd_size x 1
        policy = tf.squeeze(policy, axis=1)

        return value, policy

