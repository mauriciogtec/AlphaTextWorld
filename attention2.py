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
    REG_PENALTY = 1e-8
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
            num_blocks=2,
            l2=self.REG_PENALTY)

        self.att_memory_loc_turn = PairedAttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=2,
            l2=self.REG_PENALTY)

        self.att_memory_cmdlist_time = AttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=2,
            l2=self.REG_PENALTY)

        self.att_memory_cmdlist_turn = PairedAttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=2,
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
            num_blocks=2,
            l2=self.REG_PENALTY)

        self.att_cmd_gen_prev = AttentionEncoder(
            units=self.HIDDEN_UNITS,
            num_heads=self.ATT_HEADS,
            num_blocks=2,
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
            K = prevx.shape[0]

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
        memtpx = tf.concat([memtpx, posx], axis=-1)  # M x (dim + posfreq)
        memtpx = tf.expand_dims(memtpx, axis=0)  # 1 x M x (dim + posfreq)
        x = self.att_memory_loc_turn(
            locx, memtpx, training=training)  # 1 x (dim)
        value = self.value_head(x, training=training)  # 1
        value = tf.squeeze(value)  # ()

        # 2. pipeline for action value prediction
        x = self.att_memory_cmdlist_time(
            queryx, memx, training=training)  # M x C x dim
        memx = []  # not used again, free memory
        x = tf.transpose(x, perm=(1, 0, 2))  # C X M X dim
        posx = self.position_encodings(M, self.POSFREQS)  # M x pfreq
        posx = tf.stack([posx] * C, axis=0)  # C x M x posfreq
        x = tf.concat([x, posx], axis=-1)  # C x M x (dim + pfrq)
        x = self.att_memory_cmdlist_turn(
            queryx, x, training=training)  # C x dim
        posx = []  # so it gets collected
        tilelocx = tf.tile(locx, multiples=(C, 1))  # C x D
        x = tf.concat([x, tilelocx], axis=-1)  # C x 2D
        tilelocx = []  # free memory
        x = self.policy_head(x, training=training)  # (C)
        policy_logits = tf.clip_by_value(
            x, clip_value_min=-10, clip_value_max=10)

        output = {'value': value, 'policy_logits': policy_logits}

        # 3. pipeline for command prediction
        if training:
            # -- A. obtain vocabulary, location, and memory context
            cmdvocab2id = ents2id
            V = len(cmdvocab2id)
            # vocabx = self.encode_text(cmdvocab)
            queryx = tf.math.reduce_sum(vocabx, axis=1)  # V x dim

            memvocabx = self.att_cmd_gen_mem(
                queryx, memtpx, training=training)  # 1 x V x dim
            memvocabx = tf.squeeze(memvocabx, axis=0)  # V x dim

            # -- B. sequential decoding in teacher mode
            prevx = self.att_cmd_gen_prev(
                memvocabx, prevx, training=training)  # K X V X D
            tilelocx = tf.tile(locx, multiples=(K, 1))  # K x D
            currentx = self.embeddings(lastcmdent)  # K x D
            currentx = tf.concat([currentx, tilelocx], axis=1)  # K x 2D
            tilelocx = []  # free memory
            currentx = tf.expand_dims(currentx, axis=1)  # K x 1 x 2D
            currentx = tf.tile(currentx, multiples=(1, V, 1))  # K x V x 2D
            currentx = tf.concat([currentx, prevx], axis=-1)  # K X V X 3D
            x = tf.reshape(currentx, (-1, 3 * self.HIDDEN_UNITS))
            x = self.cmd_gen_head(x, training=training)  # (K * V) x D
            nextword_logits = tf.reshape(x, shape=(-1, V))  # K x V

            output['nextword_logits'] = nextword_logits

        return output
