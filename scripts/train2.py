import argparse
import glob
import re
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations,\
    optimizers, utils, layers, models, regularizers, initializers
import sys
import time
import ujson
import math
import os
import gc


# ----------------------
description = "Load data and train network"
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--num_consider',
                    type=int,
                    default=1000,
                    help='Number of games latest games from which to sample')
parser.add_argument('--num_data',
                    type=int,
                    default=2000,
                    help='Number of data points to sample')
parser.add_argument('--batch_size',
                    type=int,
                    default=4,
                    help='Batch size')
parser.add_argument('--ckpt_every',
                    type=int,
                    default=25,
                    help='How many batches often print and save results')
parser.add_argument('--num_epochs',
                    type=int,
                    default=2,
                    help='Epochs in these data.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.01,
                    help='Learning rate')

parser.add_argument('--cwd', default='./',
                    help='Directory from which to launch')
parser.add_argument('--model_dir', default='trained_models2/',
                    help='Directory in which to save game results')
args = parser.parse_args()
print("Arguments: ", args)

cwd = args.cwd
model_dir = args.model_dir
# num_choice = args.num_choice
num_consider = args.num_consider
num_data = args.num_data
ckpt_every = args.ckpt_every
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate

sys.path.append(cwd)
from custom_layers import *
from textutils import *
import attention2 as nn
import mctsagent as mcts
import nltk
# ----------------------

# cwd = "."

textworld_vocab = set()
with open(cwd + 'textworld_vocab.txt', 'r') as fn:
    for line in fn:
        word = line[:-1]
        textworld_vocab.add(word)

embeddingsdir = cwd + "../glove.6B/"
embedding_dim = 100
embedding_fdim = 64
embeddings, vocab = load_embeddings(
    embeddingsdir=embeddingsdir,
    embedding_dim=embedding_dim,  # try 50
    embedding_fdim=embedding_dim,
    seed=None,
    vocab=textworld_vocab)

index = np.random.permutation(range(embedding_dim))[:embedding_fdim]
embeddings = embeddings[index, :]


words = [x for x in textworld_vocab if x != "" and not re.search("[^a-z]", x)]
tags = nltk.pos_tag(words)

nouns = [x[0] for x in tags if x[1] == 'NN']
adjectives = [x[0] for x in tags if x[1] == 'JJ']

# instantiate network
network = nn.AlphaTextWorldNet(embeddings, vocab)
network(inputs={
    'memory_input': tf.constant([[0]], tf.int32),
    'cmdlist_input': tf.constant([[0]], tf.int32),
    'location_input': tf.constant([0], tf.int32),
    'cmdprev_input': tf.constant([[0]], tf.int32),
    'ents2id': {".": 0},
    'lastcmdent_input': tf.constant([0], tf.int32),
    'entvocab_input': tf.constant([[0]], tf.int32)},
    training=True)
print(network.summary())

optim = tf.optimizers.Nadam(
    learning_rate=learning_rate,
    clipnorm=45.0,
    beta_1=0.5,
    beta_2=0.7)

tstamp = math.trunc(100 * time.time())

# load model if existing
modelfiles = glob.glob("{}{}*.h5".format(cwd, model_dir))
if len(modelfiles) > 0:
    fn = max(modelfiles)
    print("Loading weights:", fn)
    network.load_weights(fn)
else:
    network.save_weights('{}trained_models2/{}.h5'.format(cwd, tstamp))


def get_batch(x, i, batch_size):
    return x[(i*batch_size):((i+1)*batch_size)]

VERBS = ["take", "cook", "go", "open", "drop", "slice",
         "eat", "prepare", "examine", "chop", "dice", "drink"]
PREPS = ["with", "from"]
UNWANTED_WORDS = ['a', 'an', 'the']


def get_entities(feedback_history):
    memory = feedback_history
    entities = set(
        ["<PAD>", "<UNK>", "<S>", "</S>"] + VERBS + PREPS)
    for entry in memory:
        entities.update(entry['entities'])
    return entities


def get_location_and_directions(feedback_history):
    locs = [x for x in feedback_history if x['is_valid'] and x['is_location']]
    loc = locs[-1] if len(locs) > 0 else "unknown"
    return loc['location'], loc['directions'], loc['entities']


def tokenize_from_cmd_template(cmd):
    words = [x for x in cmd.split() if x not in UNWANTED_WORDS]
    template = [words[0]]
    i = 1
    s = words[1]
    while i < len(words) - 1:
        if words[i + 1] not in PREPS:
            s += ' ' + words[i + 1]
            i += 1
        else:
            template.append(s)
            template.append(words[i + 1])
            s = words[i + 2]
            i += 2
    template.append(s)
    return template


def train(model, optim, data_batch):
    batch_size = len(data_batch)
    
    inputs_batch = [d['inputs'] for d in data_batch]
    cmdlist_batch = [d['cmdlist'] for d in data_batch]
    value_batch = [d['value'] for d in data_batch]
    counts_batch = [d['counts'] for d in data_batch]
    policy_batch = [np.array(x) / sum(x) for x in counts_batch]
    policy_batch = [0.98 * p + 0.02 / len(p) for p in policy_batch]
    memory_batch = [d['feedback_history'] for d in data_batch]

    # nwoutput_batch = [d['nwoutput'] for d in data_batch]  # buggy
    value_loss, policy_loss, cmdgen_loss, reg_loss = 0, 0, 0, 0
    with tf.GradientTape() as tape:
        for i in range(batch_size):
            x = inputs_batch[i]
            value = value_batch[i]
            policy = policy_batch[i]
            cmds = cmdlist_batch[i]
            memory = memory_batch[i]
            loc, dirs, ent_locs = get_location_and_directions(memory)
            cmd_tokens = [tokenize_from_cmd_template(cmd) for cmd in cmds]
            ent_locs = set(ent_locs + dirs)
            entities = get_entities(memory)
            entities.update(dirs)

            # fix unseen entities for cmds =====
            # path to the left
            # add latest command for sequential prediction
            cmdid_in_ents = [
                i for i, toks in enumerate(cmd_tokens)
                if toks[1] in entities and (len(toks) < 4 or toks[3] in entities)]
            cmds = [cmds[i] for i in cmdid_in_ents]
            cmdlist_input = x['cmdlist_input']
            cmdlist_input = [cmdlist_input[i] for i in cmdid_in_ents]
            policy = [policy[i] for i in cmdid_in_ents]
            if len(cmds) == 0:
                continue
            # fix entities to have valid directions
            ents2id = x['ents2id']
            entvocab_input = x['entvocab_input']
            for d in dirs:
                if d not in ents2id:
                    idx = len(ents2id)
                    ents2id[d] = idx
                    padlen = len(entvocab_input[0])
                    entvocab_input.append([idx] + [0] * (padlen - 1))
            # ========================================
            pad, stend, unk = ents2id['<PAD>'], ents2id['</S>'], ents2id['<UNK>']
            cmdprev_input = x['cmdprev_input']
            C, V, K = len(cmds), len(ents2id), len(cmdprev_input)
            cmdents = [[i for i in z if i != pad] for z in cmdprev_input]
            nwoutput = []
            idx_include = []
            j = 0
            for i in range(K - 1):
                if unk not in cmdents[i]:
                    if len(cmdents[i + 1]) > len(cmdents[i]):
                        j += 1
                        nwoutput.append(cmdents[i+1][j])
                    else:
                        j = 0
                        nwoutput.append(stend)
                    idx_include.append(i)
                else:
                    j = 0
            if unk not in cmdents[K - 1]:
                nwoutput.append(stend)
                idx_include.append(K - 1)
            cmdents = [cmdents[i] for i in idx_include]
            cmdprev_input = [cmdprev_input[i] for i in idx_include]
            lastcmdent_input = [x[-1] for x in cmdents]
            # ====================================

            cmdlist_input = tf.constant(cmdlist_input, tf.int32)
            memory_input = tf.constant(x['memory_input'], tf.int32)
            cmdprev_input = tf.constant(cmdprev_input, tf.int32)
            entvocab_input = tf.constant(entvocab_input, tf.int32)
            location_input = tf.constant(x['location_input'], tf.int32)
            lastcmdent_input = tf.constant(lastcmdent_input, tf.int32)
            # skip round if there's only one command
            # if len(cmdlist) < 2:
            #     continue
            # evaluate model
            C, V, K = len(cmds), len(ents2id), len(cmdprev_input)
            inputs = {'memory_input': memory_input,
                      'cmdlist_input': cmdlist_input,
                      'entvocab_input': entvocab_input,
                      'cmdprev_input': cmdprev_input,
                      'location_input': location_input,
                      'lastcmdent_input': lastcmdent_input,
                      'ents2id': ents2id}
            output = model(inputs, training=True)
            # value loss
            vhat = output['value']
            value_loss += tf.square(value - vhat)
            # policy loss
            plogits = output['policy_logits']
            phat = tf.math.softmax(plogits)
            logphat = tf.math.log(phat + 1e-12)
            policy_loss += - tf.reduce_sum(logphat * policy)

            nwlogits = output['nextword_logits']

            nwp = tf.math.softmax(nwlogits, axis=-1)
            lognwp = tf.math.log(nwp + 1e-12)
            nwoutputx = tf.one_hot(nwoutput, depth=V)
            cmdgen_loss += - tf.reduce_sum(lognwp * nwoutputx) / K

            reg_loss += tf.math.add_n(
                [l for l in model.losses
                 if not np.isnan(l.numpy())]) / batch_size
        # add losses
        value_loss /= batch_size
        policy_loss /= batch_size
        cmdgen_loss /= batch_size
        reg_loss /= batch_size
        loss = value_loss + policy_loss + cmdgen_loss + reg_loss
    # apply gradients
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    update = optim.apply_gradients(zip(gradients, variables))

    return value_loss, policy_loss, cmdgen_loss, reg_loss, loss


# Pull random games from last games
all_batchfiles = glob.glob("data/*.json")
all_batchfiles.sort(reverse=True)
all_batchfiles = all_batchfiles[:num_consider]  # exclude current

# extend current data
data = []
for datafile in all_batchfiles:
    # datafile = "data/{}.json".format(s)
    # print("Adding replay data from:", datafile)
    with open(datafile, 'r') as fn:
        d = ujson.load(fn)
        data.extend(d)

# data = data_current

# learn from winning losing or nothin
data = [x for x in data if
        (x['value'] > 0.5) or
        (-0.03 < x['value'] < 0.05) or
        (x['value'] < -0.25)]

# order data and obtain value policy and nextwords
# data = np.random.permutation(data)
data_idx = np.random.choice(np.arange(len(data)), num_data)
data = [data[i] for i in data_idx]

ndata = len(data)
num_batches = ndata // batch_size

msg = "OPTIMIZATION: epochs: {} batches: {}  total plays: {}"
print(msg.format(num_epochs, num_batches, len(data)))

iteration = 0
for e in range(num_epochs):
    mv, mp, mcg, mr, ml = 0, 0, 0, 0, 0
    for b in range(num_batches):
        data_batch = get_batch(data, b, batch_size)

        try:
            vloss, ploss, cgloss, rloss, loss = train(
                network, optim, data_batch)
        except Exception as e:
            print(e)
            continue

        msg = "Optimizing... epoch: {} batch: {:2d}, iter: {:3d}, " +\
            "vloss: {:.2f}, ploss: {:.2f}, " +\
            "cgloss: {:.2f}, rloss {:.4f}, loss {:.2f}"

        print(msg.format(
            e, b, (b + 1) * (batch_size), vloss.numpy().item(),
            ploss.numpy().item(), cgloss.numpy().item(),
            rloss.numpy().item(), loss.numpy().item()))

        M = iteration % ckpt_every
        mv += (vloss.numpy().item() - mv) / (M + 1)
        mp += (ploss.numpy().item() - mp) / (M + 1)
        mcg += (cgloss.numpy().item() - mcg) / (M + 1)
        mr += (rloss.numpy().item() - mr) / (M + 1)
        ml += (loss.numpy().item() - ml) / (M + 1)

        iteration += 1

        if iteration % ckpt_every == 0:
            tstamp = math.trunc(100 * time.time())
            wfile = "trained_models2/{}.h5".format(tstamp)
            print("saving trained weights to {}...".format(wfile))
            network.save_weights(wfile)

            msg = "".join(["Ckpt summary: vloss: {:.2f}, ploss: {:.2f}",
                           ", cgloss: {:.2f}, rloss: {:.2f}, loss: {:.2f}"])
            print(msg.format(mv, mp, mcg, mr, ml))
            mv, mp, mcg, mr, ml = 0, 0, 0, 0, 0

        gc.collect()

tstamp = math.trunc(100 * time.time())
wfile = "trained_models2/{}.h5".format(tstamp)
print("saving trained weights to {}...".format(wfile))
network.save_weights(wfile)

print(0)