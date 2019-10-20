# This code contains modified code from
# https://github.com/Lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations,\
    optimizers, utils, layers, models, regularizers, initializers
import pdb
import sys

sys.path.append("../")
import textutils


class TimeSelfAttention(models.Model):
    def __init__(self, units, dropout=0.1, l2=None, **kwargs):
        super(TimeSelfAttention, self).__init__(**kwargs)
        self.dropout = layers.Dropout(dropout)
        self.attention = layers.TimeDistributed(
            layers.Dense(
                units=1,
                kernel_regularizer=regularizers.l2(l2)))
        self.value_layer = layers.TimeDistributed(
            layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2)))

    def call(self, inputs, training=False, attentions=False):
        """
        inputs
            keys: (batch_size x timesteps x input_dim)
        returns
            output: context vector (batch_size x output_dim)
            attn: attn weights (batch_size x timesteps)
        """
        scores = tf.squeeze(self.attention(inputs), axis=2)  # batch_size x timesteps
        attn_weights = tf.math.softmax(scores, axis=1)
        attn_weights_expanded = tf.expand_dims(attn_weights, axis=1)
        values = self.value_layer(inputs)  # batch_size x timesteps x outputd
        output = tf.matmul(attn_weights_expanded, values)

        if attentions:
            return output, attn_weights
        else:
            return output

# class TimeContextPooling(models.Model):
#     def __init__(self, units, dropout=0.1, l2=None, **kwargs):
#         super(TimeSelfPooling, self).__init__(**kwargs)
#         self.dropout = layers.Dropout(dropout)
#         self.layer_norm = LayerNormalization()
#         self.attention = layers.TimeDistributed(
#             layers.Dense(
#                 units=1,
#                 use_bias=False,
#                 kernel_regularizer=regularizers.l2(l2)))
#         self.value_layer = layers.Dense(
#             units=units,
#             activation='relu',
#             kernel_regularizer=regularizers.l2(l2))

#     def call(self, inputs, contexts, training=False, attentions=False):
#         """
#         inputs
#             inputs: (pool_dim x input_dim)
#             context: (contexts_size x context_dim)
#         returns
#             output: context vector (context_size x output_dim)
#         """
#         csize, pdim = contexts.shape[0], inputs.shape[0]
#         inputs = tf.expand_dims(inputs, axis=0)  # 1 x pdim x idim
#         inputs = tf.tile(inputs, (csize, 1, 1))  # csize x pdim x idim
#         contexts = tf.expand_dims(contexts, axis=1)  # csize x 1 x cdim
#         contexts = tf.tile(contexts, (1, pdim, 1))  # csize x pdim x cdim
#         x = tf.concat([inputs, contexts], axis=1)  # csize x pdim x (i+c)dim
#         scores = tf.squeeze(self.attention(x), axis=2)  # csize x pdim
#         attn_wts = tf.math.softmax(scores, axis=1)  # csize x pdim
#         values = self.value_layer(x)  # pdim x odim
#         output = tf.matmul(attn_wts, values)  # csize x odim
#         return output


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


class ScaledDotProductAttention(layers.Layer):
    def __init__(self, dropout=0.1, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = layers.Dropout(dropout)

    def call(self, queries, keys, values, mask=None,
             training=False, attentions=False):
        """
        inputs
            q: (queries_size x timesteps x input_dim)
            keys: (keys memory_size x memory_timesteps x input_dim)
            values: (values queries_size x timesteps x output_dim)
        returns
            output: context vector (queries_size x timesteps x out_dim)
            attn: attn weights (queries_size x timesteps x out_dim)
        """
        attn = tf.matmul(queries, keys, transpose_b=True)
        scale_dim = keys.shape[2]
        attn /= np.sqrt(scale_dim)
        if mask is not None:
            attn += mask
        attn = activations.softmax(attn)
        if training:
            attn = self.dropout(attn)
        output = tf.matmul(attn, values)

        if attentions:
            return output, attn
        else:
            return output


class MultiHeadAttention(models.Model):
    def __init__(self, units, num_heads, residual=False,
                 l2=None, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.keys_dim = units // num_heads
        self.residual = residual

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        for _ in range(num_heads):
            self.qs_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        self.keys_dim,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(l2))))
            self.ks_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        self.keys_dim,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(l2))))
            self.vs_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        units=self.keys_dim,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(l2))))

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dense = layers.TimeDistributed(
            layers.Dense(
                units,
                activation="relu",
                use_bias=False,
                kernel_regularizer=regularizers.l2(l2)))
        self.dropout = layers.SpatialDropout1D(dropout)
        self.layernorm = LayerNormalization()

    def call(self, queries, keys, training=False, attentions=False):
        heads = []
        attns = []
        # keys_size = keys.shape[0]
        for i in range(self.num_heads):
            # Q = tf.expand_dims(queries, axis=0)
            Q = self.qs_layers[i](queries)
            # Q = tf.tile(Q, (keys_size, 1, 1))
            K = self.ks_layers[i](keys)
            V = self.vs_layers[i](keys)
            head, attn = self.attention(
                Q, K, V,
                attentions=True, training=training)
            heads.append(head)
            attns.append(attn)
        output = tf.concat(heads, axis=2)
        output = self.dense(output)
        if training:
            output = self.dropout(output)
        if self.residual:
            output += queries
        output = self.layernorm(output)
        return output


class MultiHeadSelfAttention(models.Model):
    #
    def __init__(self, units, num_heads=1, axis=2,
                 residual=False, l2=None, dropout=0.1, **kwargs):
        #
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        assert units % num_heads == 0
        self.num_heads = num_heads
        self.keys_dim = units // num_heads
        self.axis = axis
        self.residual = residual

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        for _ in range(num_heads):
            self.qs_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        self.keys_dim,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(l2))))
            self.ks_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        self.keys_dim,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(l2))))
            self.vs_layers.append(
                layers.TimeDistributed(
                    layers.Dense(
                        self.keys_dim,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(l2))))

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dense = layers.TimeDistributed(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2)))
        self.dropout = layers.SpatialDropout1D(dropout)
        self.layernorm = LayerNormalization()

    def call(self, inputs, training=False, attentions=False):
        heads = []
        attns = []
        for i in range(self.num_heads):
            Q = self.qs_layers[i](inputs)
            K = self.ks_layers[i](inputs)
            V = self.vs_layers[i](inputs)
            head, attn = self.attention(
                Q, K, V, attentions=True, training=training)
            heads.append(head)
            attns.append(attn)

        output = tf.concat(heads, axis=self.axis)
        output = self.dense(output)
        if training:
            output = self.dropout(output)
        if self.residual:
            output += inputs
        output = self.layernorm(output)

        if attentions:
            return output, attns
        else:
            return output


class SelfAttentionEncoder(models.Model):
    """Transforms text to memory and output vectors"""
    def __init__(self, units, num_heads, num_blocks,
                 l2=None, **kwargs):
        #
        super(SelfAttentionEncoder, self).__init__(**kwargs)
        self.conv1 = layers.Conv1D(
            filters=units,
            kernel_size=1,
            padding='same',
            kernel_regularizer=regularizers.l2(l2))
        self.layernorm1 = LayerNormalization()
        #
        self.relu1 = layers.Activation('relu')
        self.conv2 = layers.Conv1D(
            filters=units,
            kernel_size=3,
            padding='same',
            kernel_regularizer=regularizers.l2(l2))
        self.layernorm2 = LayerNormalization()
        self.relu2 = layers.Activation('relu')
        #
        self.att_blocks = [
            MultiHeadSelfAttention(
                units=units,
                num_heads=num_heads,
                residual=True,
                axis=2, l2=l2)
            for i in range(num_blocks)]

    def call(self, inputs, training=None):
        x = inputs
        x = self.conv1(x)
        x = self.layernorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.layernorm2(x)
        x = self.relu2(x)
        for block in self.att_blocks:
            x = block(x, training=training)
        return x


class AttentionEncoder(models.Model):
    """Transforms text to memory and output vectors"""
    def __init__(self, units, num_heads, num_blocks, l2=None, **kwargs):
        #
        super(AttentionEncoder, self).__init__(**kwargs)
        #
        self.att_blocks = [
            MultiHeadAttention(
                units=units,
                num_heads=num_heads,
                residual=True,
                l2=l2)
            for i in range(num_blocks)]

    def call(self, queries, keys, training=False):
        x = queries
        for block in self.att_blocks:
            x = block(x, keys, training=training)
        return x


class DenseHead(models.Model):
    """Trained to predict a singlue value"""
    def __init__(self, hidden_units, 
                 residual=False, dropout=0.25, l2=None, **kwargs):
        super(DenseHead, self).__init__(**kwargs)
        self.dense_1 = layers.Dense(
            units=hidden_units,
            kernel_regularizer=regularizers.l2(l2),
            activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.dense_2 = layers.Dense(
            units=1,
            kernel_regularizer=regularizers.l2(l2))
        self.residual = residual

    def call(self, inputs, training=None):
        x = inputs
        x = self.dense_1(x)
        if training:
            x = self.dropout(x)
        if self.residual:
            x += inputs
        x = self.dense_2(x)
        x = tf.squeeze(x)
        return x


class AlphaTextWorldNet(models.Model):
    """
    Learn to play from memory
    """
    REG_PENALTY = 1e-5
    HIDDEN_UNITS = 64
    ATT_HEADS = 4
    POSFREQS = 16

    def __init__(self,  embeddings, vocab, **kwargs):
        super(AlphaTextWorldNet, self).__init__(**kwargs)

        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for i, w in self.word2id.items()}

        embedding_dim, vocab_size = embeddings.shape
        self.embeddings = layers.Embedding(
            input_dim=vocab_size,
            input_length=None,
            output_dim=embedding_dim,
            embeddings_initializer=initializers.Constant(embeddings),
            trainable=True,
            name="embeddings")

        self.memory_encoder = SelfAttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=2,
            l2=self.REG_PENALTY,
            name="memory_encoder")

        self.cmd_encoder = SelfAttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=1,
            l2=self.REG_PENALTY,
            name="cmd_encoder")

        self.attention_encoder = AttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=2,
            l2=self.REG_PENALTY,
            name="att_encoder")

        self.memory_time_encode = TimeSelfAttention(
            units=self.HIDDEN_UNITS,
            l2=self.REG_PENALTY,
            name="value_time_encode")

        self.memory_turn_encode = TimeSelfAttention(
            units=self.HIDDEN_UNITS + self.POSFREQS,
            l2=self.REG_PENALTY,
            name="value_turn_encode")

        self.value_head = DenseHead(
            hidden_units=self.HIDDEN_UNITS,
            dropout=0.5,
            l2=self.REG_PENALTY,
            name="value_head")

        self.cmd_turn_encode = TimeSelfAttention(
            units=self.HIDDEN_UNITS + self.POSFREQS,
            l2=self.REG_PENALTY,
            name="cmd_turn_encode")

        self.policy_head = DenseHead(
            hidden_units=self.HIDDEN_UNITS + self.POSFREQS,
            dropout=0.5,
            l2=self.REG_PENALTY,
            name="policy_head")

    def position_encodings(self, depth, freqs):
        ans = []
        for r in range(freqs):
            scale = 1000**(r / freqs)
            x = np.arange(1 - depth, 1) / scale
            x = np.sin(x) if r % 2 == 0 else np.cos(x)
            ans.append(x)
        ans = np.vstack(ans)
        ans = ans.transpose()
        ans = tf.constant(ans, tf.float32)
        return ans

    def encode_text(self, textlist):
        """common pattern: embed -> encode"""
        x = textutils.text2tensor(textlist, self.word2id, max_token_length=32)
        x = self.embeddings(x)
        return x

    # @tf.function # faster eval slower backprop
    def call(self, inputs, training=False):
        memory, cmdlist = inputs

        # obtain embeddings and self-encode commands amd memory
        memx = self.encode_text(memory)
        memx = self.memory_encoder(memx, training=training)
        cmdx = self.encode_text(cmdlist)
        cmdx = self.cmd_encoder(cmdx, training=training)
        M = memx.shape[0]
        C = cmdx.shape[0]

        # output a vector of context for each memory key
        queryx = tf.math.reduce_mean(cmdx, axis=1)
        queryx = tf.expand_dims(queryx, axis=0)
        queryx = tf.tile(queryx, (M, 1, 1))

        attx = self.attention_encoder(queryx, memx, training=training)

        # add position encodings
        posx = self.position_encodings(depth=M, freqs=self.POSFREQS)
        posx = tf.expand_dims(posx, axis=1)
        posx = tf.tile(posx, (1, C, 1))
        attx = tf.concat([attx, posx], axis=2)

        # value of a state
        value = self.memory_time_encode(memx, training=training)
        value = tf.squeeze(value, axis=1)  # cmds x hidden_dim
        posx = self.position_encodings(depth=M, freqs=self.POSFREQS)
        value = tf.concat([value, posx], axis=1)
        value = tf.expand_dims(value, axis=0)
        value = self.memory_turn_encode(value, training=training)
        value = tf.squeeze(value, axis=1)
        value = self.value_head(value, training=training)

        # policy
        attx = tf.transpose(attx, perm=(1, 0, 2))
        policy_logits = self.cmd_turn_encode(attx, training=training)
        policy_logits = tf.squeeze(policy_logits, axis=1)
        policy_logits = self.policy_head(policy_logits, training=training)

        return value, policy_logits


def load_network(embeddings, vocab, path_to_weights):
    model = AlphaTextWorldNet(embeddings, vocab)
    cmdlist = [".", ".."]
    memory = ["hi there", "how", "are you holding on"]
    training = True
    initrun = model((memory, cmdlist), training=training)
    model.load_weights(path_to_weights)
    return model


# textworld_vocab = set()
# with open('../TextWorld/montecarlo/vocab.txt', 'r') as fn:
#     for line in fn:
#         word = line[:-1]
#         textworld_vocab.add(word)

# embeddings, vocab = textutils.load_embeddings(
#     embeddingsdir="../../glove.6B/",
#     embedding_dim=300,  # try 50
#     vocab=textworld_vocab)
# np.random.seed(110104)
# index = np.random.permutation(range(300))[:256]
# embeddings = embeddings[index, :]