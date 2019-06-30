# This code contains modified code from
# https://github.com/Lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations,\
    optimizers, utils, layers, models, regularizers, initializers
import pdb
import sys
import re
from collections import deque
from custom_layers import *
from textutils import *


class AlphaTextWorldNet(models.Model):
    """
    Learn to play from memory
    """
    REG_PENALTY = 1e-6
    KSIZE = 3
    HIDDEN_UNITS = 64
    ATT_HEADS = 4
    POSFREQS = 16
    MAX_CMD_LEN = 6

    def __init__(self,  embeddings, vocab, **kwargs):
        super(AlphaTextWorldNet, self).__init__(**kwargs)

        self.vocab = vocab
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.verbs = ["take", "cook", "go", "open", "drop",
                      "eat", "prepare", "examine", "chop", "dice"]
        self.adverbs = ["with", "from"]
        self.unnecessary_words = ['a', 'an', 'the']

        embedding_dim, vocab_size = embeddings.shape
        self.embeddings = layers.Embedding(
            input_dim=vocab_size,
            input_length=None,
            output_dim=embedding_dim,
            embeddings_initializer=initializers.Constant(embeddings),
            trainable=False)

        self.lfe_memory = LocalFeaturesExtractor(
            filters=self.HIDDEN_UNITS,
            kernel_size=self.KSIZE,
            num_blocks=5,
            l2=self.REG_PENALTY)

        self.lfe_cmdlist = LocalFeaturesExtractor(
            filters=self.HIDDEN_UNITS,
            kernel_size=self.KSIZE,
            num_blocks=2,
            l2=self.REG_PENALTY)

        self.att_memory_loc_time = AttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=5,
            l2=self.REG_PENALTY)

        self.att_memory_loc_turn = PairedAttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=5,
            l2=self.REG_PENALTY)

        self.att_memory_cmdlist_time = AttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=5,
            l2=self.REG_PENALTY)

        self.att_memory_cmdlist_turn = PairedAttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=5,
            l2=self.REG_PENALTY)

        self.value_head = DenseHead(
            hidden_units=self.HIDDEN_UNITS,
            l2=self.REG_PENALTY)

        self.policy_head = DenseHead(
            hidden_units=self.HIDDEN_UNITS,
            l2=self.REG_PENALTY)

        self.cmd_gen_head = DenseHead(
            hidden_units=self.HIDDEN_UNITS,
            l2=self.REG_PENALTY)

        self.att_cmd_gen_mem = AttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=4,
            l2=self.REG_PENALTY)

        self.att_cmd_gen_prev = AttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=5,
            l2=self.REG_PENALTY)

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
        x = text2tensor(textlist, self.word2id)
        x = self.embeddings(x)
        return x

    def get_location(self, memory):
        N = len(memory)
        i = N - 1
        while i >= 0:
            x = re.search("= ([A-Za-z]+) =", memory[i])
            if x is not None:
                return x.group(1).lower()
            i -= 1
        return 'unknown'

    def split_from_cmd_template(self, cmd):
        words = [x for x in cmd.split() if x not in self.unnecessary_words]
        template = [words[0]]
        i = 1
        s = words[1]
        while i < len(words) - 1:
            if words[i + 1] not in self.adverbs:
                s += ' ' + words[i + 1]
                i += 1
            else:
                template.append(s)
                template.append(words[i + 1])
                s = words[i + 2]
                i += 2
        template.append(s)
        return template

    # @tf.function # faster eval slower backprop
    def call(self, inputs, training=False):
        memory = inputs['memory_input']
        cmdlist = inputs['cmdlist_input']
        location = inputs['location_input']

        if training:
            ents2id = inputs['ents2id']
            entvocab = inputs['entvocab_input']
            cmdprev = inputs['cmdprev_input']
            lastcmdent = inputs['lastcmdent_input']

        # obtain embeddings and self-encode commands amd memory
        memx = self.embeddings(memory)
        cmdx = self.embeddings(cmdlist)
        locx = self.embeddings(location)

        if training:
            vocabx = self.embeddings(entvocab)
            prevx = self.embeddings(cmdprev)

        M = memx.shape[0]
        C = cmdx.shape[0]

        memx = self.lfe_memory(memx)  # M x T x dim
        cmdx = self.lfe_cmdlist(cmdx)  # C x Tc x dim
        queryx = tf.math.reduce_sum(cmdx, axis=1)  # C x dim

        # 1. pipeline for value prediction
        memtpx = self.att_memory_loc_time(
            locx, memx, training=training)  # M x 1 x dim
        memtpx = tf.squeeze(memtpx, axis=1)  # M X dim
        posx = self.position_encodings(M, self.POSFREQS)  # M x pfreq
        memtpx = tf.concat([memtpx, posx], axis=1)  # M x (dim + posfreq)
        memtpx = tf.expand_dims(memtpx, axis=0)  # 1 x M x (dim + posfreq)
        x = self.att_memory_loc_turn(
            locx, memtpx, training=training)  # 1 x (dim)
        x += locx  # 1 x dim
        value = self.value_head(x, training=training)  # 1
        value = tf.squeeze(value)  # ()

        # 2. pipeline for action value prediction
        x = self.att_memory_cmdlist_time(
            queryx, memx, training=training)  # M x C x dim
        x = tf.transpose(x, perm=(1, 0, 2))  # C X M X dim
        posx = self.position_encodings(M, self.POSFREQS)  # M x pfreq
        posx = tf.stack([posx] * C, axis=0)  # C x M x posfreq
        x = tf.concat([x, posx], axis=-1)  # C x M x (dim + pfrq)
        x = self.att_memory_cmdlist_turn(
            queryx, x, training=training)  # C x dim
        x += locx  # C x dim
        x = self.policy_head(x, training=training)  # (C)
        policy_logits = tf.clip_by_value(
            x, clip_value_min=-10, clip_value_max=10)

        output = {'value': value, 'policy_logits': policy_logits}

        # 3. pipeline for command prediction
        if training:
            # -- A. obtain vocabulary, location, and memory context
            # cmdvocab = ["<PAD>", "<UNK>", "<S>", "</S>"] +\
            #      self.verbs + self.adverbs + noun_phrases(memory)
            # cmdvocab2id = {x: i for i, x in enumerate(cmdvocab)}
            cmdvocab2id = ents2id
            V = len(cmdvocab2id)
            # vocabx = self.encode_text(cmdvocab)
            queryx = tf.math.reduce_sum(vocabx, axis=1)  # V x dim
            queryx += locx

            memvocabx = self.att_cmd_gen_mem(
                queryx, memtpx, training=training)  # 1 x V x dim
            memvocabx = tf.squeeze(memvocabx, axis=0)  # V x dim

            # -- B. sequential decoding in teacher mode
            # cmds_deque = deque(cmdlist)
            # cmd = cmds_deque.popleft()
            # nextword_logits = []
            currentx = self.embeddings(lastcmdent)
            currentx = tf.expand_dims(currentx, axis=1)
            prevx = self.att_cmd_gen_prev(
                memvocabx, prevx, training=training)  # NPC X V X D
            contextx = prevx + currentx   # NPC X V X D
            x = tf.reshape(contextx, (-1, self.HIDDEN_UNITS))
            x = self.cmd_gen_head(x, training=training)  # (NPC*N) x D
            nextword_logits = tf.reshape(x, (-1, V))  # NPC x V
   
            # nextword_tokens = []
            # while len(cmds_deque) > 0:
            #     cmd_comps = self.split_from_cmd_template(cmd) + ['</S>']
            #     cmd_tokens = [get_word_id(x, cmdvocab2id) for x in cmd_comps]
            #     sentence_x = []  # (ntokens + 1) x [dim]
            #     logits = []  # (ntokens) x [V]
            #     for phrase in (['<S>'] + cmd_comps):
            #         x = self.encode_text([phrase])
            #         x = tf.squeeze(tf.reduce_sum(x, axis=1))  # dim
            #         sentence_x.append(x)
            #     for i in range(len(cmd_comps)):
            #         prevx = sentence_x[:(i + 1)]
            #         prevx = tf.stack(prevx, axis=0)  # nprev x dim
            #         prevx = tf.expand_dims(prevx, axis=0)  # 1 x V x dim
            #         prevx = self.att_cmd_gen_prev(
            #             vocabx, prevx, training=training)  # 1 x V x dim
            #         prevx = tf.squeeze(prevx, axis=0)  # V x dim
            #         x = memvocabx + prevx + locx + sentence_x[-1]  # V x dm
            #         x = self.cmd_gen_head(x)  # (V)
            #         logits.append(x)
            # logits = tf.stack(logits, axis=0)  # toks X V
            # nextword_logits.append(logits)  # C x [toks(c) X V]
            # nextword_tokens.append(cmd_tokens)  # C x [toks(c)]
            # cmd = cmds_deque.pop()

            output['nextword_logits'] = nextword_logits
            # output['nextword_tokens'] = nextword_tokens
            # output['cmdvocab'] = cmdvocab

        return output


# def load_network(embeddings, vocab, path_to_weights):
#     model = AlphaTextWorldNet(embeddings, vocab)
#     cmdlist = [".", ".."]
#     memory = ["hi there", "how", "are you holding on"]
#     training = True
#     initrun = model((memory, cmdlist), training=training)
#     model.load_weights(path_to_weights)
#     return model

# if __name__ == "__main__":

#     pwd = '/home/mauriciogtec/'
#     textworld_vocab = set()
#     with open(pwd + 'Github/TextWorld/montecarlo/vocab.txt', 'r') as fn:
#         for line in fn:
#             word = line[:-1]
#             textworld_vocab.add(word)

#     embedding_dim = 100
#     embedding_dim_trim = 64
#     embeddings, vocab = load_embeddings(
#         embeddingsdir=(pwd + "glove.6B/"),
#         embedding_dim=embedding_dim,  # try 50
#         vocab=textworld_vocab)
#     np.random.seed(110104)

#     index = np.random.permutation(range(embedding_dim))[:embedding_dim_trim]
#     embeddings = embeddings[index, :]

#     model = AlphaTextWorldNet(embeddings, vocab)

#     memory = ["You are hungry! Let's cook a delicious meal. Check the cookbook in the kitchen for the recipe. Once done, enjoy your meal!\n\n-", "\n-= Backyard =-\nWell, here we are in a backyard.\n\nYou scan the room, seeing a patio table. The patio table is stylish. The patio table appears to be empty. You can see a patio chair. The patio chair is stylish. However, the patio chair, like an empty patio chair, has nothing on it. Hopefully, this discovery doesn't ruin your TextWorld experience! A closed BBQ is in the room.\n\nThere is an open barn door leading east. There is an open sliding patio door leading south. You don't like doors? Why not try going north, that entranceway is not blocked by one.\n\n", '\n-= Garden =-\nYou arrive in a garden. A normal kind of place.\n\n\n\nYou need an exit without a door? You should try going south.\n\nThere is a parsley and a white onion on the floor.\n', "\n-= Backyard =-\nWell, here we are in a backyard.\n\nYou scan the room, seeing a patio table. The patio table is stylish. The patio table appears to be empty. You can see a patio chair. The patio chair is stylish. However, the patio chair, like an empty patio chair, has nothing on it. Hopefully, this discovery doesn't ruin your TextWorld experience! A closed BBQ is in the room.\n\nThere is an open barn door leading east. There is an open sliding patio door leading south. You don't like doors? Why not try going north, that entranceway is not blocked by one.\n\n", "\n-= Kitchen =-\nLook around you. Take it all in. It's not every day someone gets to be in a kitchen.\n\nYou make out an opened fridge. The fridge is empty! This is the worst thing that could possibly happen, ever! You can make out a closed oven in the room. Were you looking for a table? Because look over there, it's a table. The table is massive. But the thing is empty. What, you think everything in TextWorld should have stuff on it? You can see a counter. The counter is vast. On the counter you see a cookbook and a knife. Look over there! a stove. But the thing is empty.\n\nThere is an open plain door leading east. There is an open sliding patio door leading north. There is an exit to the south. Don't worry, there is no door. There is an exit to the west.\n\n", 'You open the copy of "Cooking: A Modern Approach (3rd Ed.)" and start reading:\n\nRecipe #1\n---------\nGather all following ingredients and follow the directions to prepare this tasty meal.\n\nIngredients:\n  chicken leg\n\nDirections:\n  grill the chicken leg\n  prepare meal\n\n\n', "\n-= Corridor =-\nYou've just shown up in a corridor.\n\n\n\nYou don't like doors? Why not try going east, that entranceway is not blocked by one. There is an exit to the north. Don't worry, there is no door. You need an exit without a door? You should try going west.\n\n", "\n-= Bedroom =-\nYou are in a bedroom. A typical one.\n\nOh wow! Is that what I think it is? It is! It's a bed. I guess it's true what they say, if you're looking for a bed, go to TextWorld. The bed is large. But oh no! there's nothing on this piece of trash. Oh! Why couldn't there just be stuff on it?\n\nThere is an exit to the south.\n\n", "\n-= Corridor =-\nYou've just shown up in a corridor.\n\n\n\nYou don't like doors? Why not try going east, that entranceway is not blocked by one. There is an exit to the north. Don't worry, there is no door. You need an exit without a door? You should try going west.\n\n", "\n-= Bedroom =-\nYou are in a bedroom. A typical one.\n\nOh wow! Is that what I think it is? It is! It's a bed. I guess it's true what they say, if you're looking for a bed, go to TextWorld. The bed is large. But oh no! there's nothing on this piece of trash. Oh! Why couldn't there just be stuff on it?\n\nThere is an exit to the south.\n\n", "\n-= Corridor =-\nYou've just shown up in a corridor.\n\n\n\nYou don't like doors? Why not try going east, that entranceway is not blocked by one. There is an exit to the north. Don't worry, there is no door. You need an exit without a door? You should try going west.\n\n", "\n-= Bathroom =-\nWell how about that, you are in a place we're calling a bathroom. You decide to just list off a complete list of everything you see in the room, because hey, why not?\n\nYou make out a toilet. The toilet is white. The toilet appears to be empty. What's the point of an empty toilet?\n\nYou don't like doors? Why not try going east, that entranceway is not blocked by one.\n\n", "\n-= Corridor =-\nYou've just shown up in a corridor.\n\n\n\nYou don't like doors? Why not try going east, that entranceway is not blocked by one. There is an exit to the north. Don't worry, there is no door. You need an exit without a door? You should try going west.\n\n", "\n-= Bathroom =-\nWell how about that, you are in a place we're calling a bathroom. You decide to just list off a complete list of everything you see in the room, because hey, why not?\n\nYou make out a toilet. The toilet is white. The toilet appears to be empty. What's the point of an empty toilet?\n\nYou don't like doors? Why not try going east, that entranceway is not blocked by one.\n\n", "\n-= Corridor =-\nYou've just shown up in a corridor.\n\n\n\nYou don't like doors? Why not try going east, that entranceway is not blocked by one. There is an exit to the north. Don't worry, there is no door. You need an exit without a door? You should try going west.\n\n", "\n-= Kitchen =-\nLook around you. Take it all in. It's not every day someone gets to be in a kitchen.\n\nYou make out an opened fridge. The fridge is empty! This is the worst thing that could possibly happen, ever! You can make out a closed oven in the room. Were you looking for a table? Because look over there, it's a table. The table is massive. But the thing is empty. What, you think everything in TextWorld should have stuff on it? You can see a counter. The counter is vast. On the counter you see a cookbook and a knife. Look over there! a stove. But the thing is empty.\n\nThere is an open plain door leading east. There is an open sliding patio door leading north. There is an exit to the south. Don't worry, there is no door. There is an exit to the west.\n\n", 'You fried the chicken wing.\n\n', 'The recipe requires a grilled chicken leg.\n', 'The recipe requires a grilled chicken leg.\n', 'The recipe requires a grilled chicken leg.\n', 'The recipe requires a grilled chicken leg.\n']

#     cmdlist = ['cook chicken leg with oven', 'cook chicken leg with stove', 'cook chicken wing with oven', 'cook chicken wing with stove', 'eat chicken wing', 'go east', 'go north', 'go south', 'go west', 'prepare meal', 'take cookbook from counter', 'take knife from counter']

#     training = True
#     inputs = {'memory': memory, 'cmdlist': cmdlist}
#     initrun = model(inputs, training=training)
#     print(0)


# # OLD CODE
# # class LayerNormalization(layers.Layer):
# #     def __init__(self, eps=1e-6, **kwargs):
# #         super(LayerNormalization, self).__init__(**kwargs)
# #         self.eps = eps

# #     def build(self, input_shape):
# #         self.gamma = self.add_weight(
# #             name='gamma',
# #             shape=input_shape[-1],
# #             initializer='ones',
# #             trainable=True)
# #         self.beta = self.add_weight(
# #             name='beta',
# #             shape=input_shape[-1],
# #             initializer='zeros',
# #             trainable=True)

# #     def call(self, x):
# #         mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
# #         std = tf.math.reduce_std(x, axis=-1, keepdims=True)
# #         return self.gamma * (x - mean) / (std + self.eps) + self.beta

# #     def compute_output_shape(self, input_shape):
# #         return input_shape


# # class TimePooling(models.Model):
# #     def __init__(self, units, dropout=0.1, l2=None, **kwargs):
# #         super(TimePooling, self).__init__(**kwargs)
# #         self.layer_norm = LayerNormalization()
# #         self.dropout = layers.Dropout(dropout)
# #         self.att_wts = layers.TimeDistributed(
# #             layers.Dense(
# #                 units=1,
# #                 kernel_regularizer=regularizers.l2(l2)))
# #         self.value_layer = layers.TimeDistributed(
# #             layers.Dense(
# #                 units=units,
# #                 activation='relu',
# #                 kernel_regularizer=regularizers.l2(l2)))

# #     def call(self, inputs, training=False, attentions=False):
# #         """
# #         inputs
# #             inputs: (batch_size x pool_dim x input_dim)
# #         returns
# #             outputs: (batch_size x output_dim)
# #         """
# #         logits = self.att_wts(inputs)  # bsize x pdim x 1
# #         attn_wts = tf.math.softmax(logits, axis=1)  # bsize x pdim x 1
# #         attn_wts = tf.transpose(attn_wts, perm=(0, 2, 1))  # bsize x 1 x pdim
# #         values = self.value_layer(inputs)  # bsize x pdim x odim
# #         output = tf.matmul(attn_wts, values)  # bsize x 1 x odim
# #         output = tf.squeeze(output, axis=1)  # bsize x odim
# #         output = self.layer_norm(output)  # bsize x odim
# #         if training:
# #             output = self.dropout(output)  # bsize x odim
# #         return output


# # class IdentityBlock(models.Model):
# #     """A residual block part of the encoder, similar to ResNet and AlphaZero"""
# #     def __init__(self, filters, kernel_size, l2=None, **kwargs):
# #         super(IdentityBlock, self).__init__(**kwargs)
# #         self.layer_norm_1 = LayerNormalization()
# #         self.relu_1 = layers.Activation('relu')
# #         self.conv_1 = layers.Conv1D(
# #             filters=filters,
# #             kernel_size=kernel_size,
# #             padding='same',
# #             kernel_regularizer=regularizers.l2(l2))
# #         self.layer_norm_2 = layers.LayerNormalization()
# #         self.relu_2 = layers.Activation('relu')
# #         self.conv_2 = layers.Conv1D(
# #             filters=filters,
# #             kernel_size=kernel_size,
# #             padding='same',
# #             kernel_regularizer=regularizers.l2(l2))
# #         self.add_layer = layers.Add()

# #     def call(self, inputs):
# #         x = self.layer_norm_1(inputs)
# #         x = self.relu_1(x)
# #         x = self.conv_1(x)
# #         x = self.layer_norm_2(x)
# #         x = self.relu_2(x)
# #         x = self.conv_2(x)
# #         x = self.add_layer([x, inputs])
# #         return x


# # class LocalFeaturesExtractor(models.Model):
# #     """Transforms text to memory and output vectors"""
# #     def __init__(self, filters, kernel_size,
# #                  num_blocks, l2=None, **kwargs):
# #         super(LocalFeaturesExtractor, self).__init__(**kwargs)
# #         self.layer_norm_1 = LayerNormalization()
# #         self.relu_1 = layers.Activation('relu')
# #         self.conv_1 = layers.Conv1D(
# #             filters=filters,
# #             kernel_size=1,
# #             padding='same',
# #             kernel_regularizer=regularizers.l2(l2))
# #         self.blocks = [
# #             IdentityBlock(
# #                 filters=filters,
# #                 kernel_size=3,
# #                 l2=l2)
# #             for i in range(num_blocks)]

# #     def call(self, inputs):
# #         x = self.layer_norm_1(inputs)
# #         x = self.relu_1(x)
# #         x = self.conv_1(x)
# #         for block in self.blocks:
# #             x = block(x)
# #         return x


# # class ScaledDotProductAttention(models.Model):
# #     def __init__(self, dropout=0.1, **kwargs):
# #         super(ScaledDotProductAttention, self).__init__(**kwargs)
# #         self.dropout = layers.Dropout(dropout)

# #     def call(self, queries, keys, values, mask=None,
# #              training=False, attentions=False):
# #         """
# #         inputs
# #             queries: (queries_size x pool_dim)
# #             keys: (keys_size x pool_dim)
# #             values: (keys_size x output_dim)
# #         returns
# #             output: (queries_size x output_dim)
# #         """
# #         attn = tf.matmul(queries, keys, transpose_b=True)  # qsize x ksize
# #         attn /= np.sqrt(keys.shape[-1])
# #         if mask is not None:
# #             attn += mask
# #         attn = tf.math.softmax(attn, axis=1)  # qsize x ksize
# #         if training:
# #             attn = self.dropout(attn)
# #         output = tf.matmul(attn, values)  # qsize x odim

# #         return output


# # class MultiHeadAttention(models.Model):
# #     def __init__(self, units, num_heads, residual=False,
# #                  l2=None, dropout=0.1, **kwargs):
# #         super(MultiHeadAttention, self).__init__(**kwargs)
# #         self.num_heads = num_heads
# #         self.units_per_head = units // num_heads
# #         self.residual = residual

# #         self.qs_layers = []
# #         self.ks_layers = []
# #         self.vs_layers = []

# #         for _ in range(num_heads):
# #             self.qs_layers.append(
# #                 layers.Dense(
# #                     self.units_per_head,
# #                     use_bias=False,
# #                     kernel_regularizer=regularizers.l2(l2)))
# #             self.ks_layers.append(
# #                 layers.Dense(
# #                     self.units_per_head,
# #                     use_bias=False,
# #                     kernel_regularizer=regularizers.l2(l2)))
# #             self.vs_layers.append(
# #                 layers.Dense(
# #                     use_bias=False,
# #                     units=self.units_per_head,
# #                     kernel_regularizer=regularizers.l2(l2)))

# #         self.attention = ScaledDotProductAttention(dropout=dropout)
# #         self.dense = layers.Dense(
# #             units,
# #             activation="relu",
# #             kernel_regularizer=regularizers.l2(l2))
# #         self.dropout = layers.Dropout(dropout)
# #         self.layernorm = LayerNormalization()

# #     def call(self, queries, keys=None, training=False):
# #         """
# #         inputs
# #             queries: (queries_size x queries_dim)
# #             keys: (keys_size x keys_dim)
# #         returns
# #             output: (queries_size x output_dim)
# #         """
# #         heads = []
# #         if keys is None:
# #             keys = queries

# #         for i in range(self.num_heads):
# #             Q = self.qs_layers[i](queries)  # qsize x hdim
# #             K = self.ks_layers[i](keys)  # ksize x hdim
# #             V = self.vs_layers[i](keys)  # ksize x hdim
# #             head = self.attention(Q, K, V, training=training)  # qsize x hdim
# #             heads.append(head)

# #         output = tf.concat(heads, axis=1)  # qsize x units
# #         output = self.dense(output)  # qsize x units
# #         if training:
# #             output = self.dropout(output)
# #         if self.residual:
# #             output += queries
# #         output = self.layernorm(output)
# #         return output


# # class AttentionEncoder(models.Model):
# #     """Transforms text to memory and output vectors"""
# #     def __init__(self, units, num_heads, num_blocks,
# #                  residual=True, l2=None, **kwargs):
# #         #
# #         super(AttentionEncoder, self).__init__(**kwargs)
# #         self.residual = residual
# #         #  note: units here shouldnt be free parameter, its always like input
# #         self.blocks = [
# #             MultiHeadAttention(
# #                 units=units,
# #                 num_heads=num_heads,
# #                 residual=self.residual,
# #                 l2=l2)
# #             for i in range(num_blocks)]

# #     def call(self, queries, keys=None, training=False):
# #         """
# #         inputs
# #             queries: (queries_size x queries_dim)
# #             keys: (keys_size x keys_dim)
# #         returns
# #             output: (queries_size x queries_dim)
# #         """
# #         x = queries
# #         for block in self.blocks:
# #             x = block(x, keys, training=training)
# #         return x


# # class DenseHead(models.Model):
# #     """Trained to predict a singlue value"""
# #     def __init__(self, hidden_units,
# #                  residual=False, dropout=0.25, l2=None, **kwargs):
# #         super(DenseHead, self).__init__(**kwargs)
# #         self.dense_1 = layers.Dense(
# #             units=hidden_units,
# #             kernel_regularizer=regularizers.l2(l2),
# #             activation='relu')
# #         self.dropout = layers.Dropout(dropout)
# #         self.dense_2 = layers.Dense(
# #             units=1,
# #             kernel_regularizer=regularizers.l2(l2))
# #         self.residual = residual

# #     def call(self, inputs, training=None):
# #         """
# #         inputs
# #             inputs: (inputs_size x input_dim)
# #         returns
# #             output: (inputs_size)
# #         """
# #         x = inputs
# #         x = self.dense_1(x)  # isize x hdim
# #         if training:
# #             x = self.dropout(x)
# #         if self.residual:
# #             x += inputs
# #         x = self.dense_2(x)  # isize x 1
# #         x = tf.squeeze(x, axis=1)  # isize
# #         return x
