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

# ----------------------
description = "Load data and train network"
parser = argparse.ArgumentParser(description=description)
# parser.add_argument('gamefile',
#                     type=str,
#                     help='Number of games to be played.')
# parser.add_argument('--subtrees',
#                     type=int, default=50,
#                     help='Subtrees to spawn.')
# parser.add_argument('--subtree_depth',
#                     type=int, default=10,
#                     help='Max depth of search trees.')
# parser.add_argument('--max_steps',
#                     type=int, default=25,
#                     help='Max number of steps per game. Defaults to 100.')
# parser.add_argument('--min_time',
#                     type=float, default=60,
#                     help=''.join(['Min time playing. If a game ends sooner',
#                                   ', it will play another episode.']))
parser.add_argument('--cwd', default='./',
                    help='Directory from which to launch')
parser.add_argument('--model_dir', default='trained_models/',
                    help='Directory in which to save game results')
args = parser.parse_args()
print("Arguments: ", args)

cwd = args.cwd
model_dir = args.model_dir

sys.path.append(cwd)
from custom_layers import *
from textutils import *
import attention as nn
import mctsagent as mcts
import nltk
# ----------------------

path = '/home/mauriciogtec/'
textworld_vocab = set()
with open(path + 'Github/TextWorld/montecarlo/vocab.txt', 'r') as fn:
    for line in fn:
        word = line[:-1]
        textworld_vocab.add(word)

words = [x for x in textworld_vocab if x != "" and not re.search("[^a-z]", x)]
tags = nltk.pos_tag(words)

nouns = [x[0] for x in tags if x[1] == 'NN']
adjectives = [x[0] for x in tags if x[1] == 'JJ']

embedding_dim = 100
embedding_fdim = 64
embeddings, vocab = load_embeddings(
    embeddingsdir="/home/mauriciogtec/glove.6B/",
    embedding_dim=embedding_dim,  # try 50
    embedding_fdim=embedding_dim,
    seed=None,
    vocab=textworld_vocab)
index = np.random.permutation(range(embedding_dim))[:embedding_fdim]
embeddings = embeddings[index, :]

# with open("./final_vocab2.txt", "w") as fn:
#     fn.write("\n".join(vocab))

# instantiate network
network = nn.AlphaTextWorldNet(embeddings, vocab)
network(inputs={
    'memory_input': tf.constant([[0]], tf.int32),
    'cmdlist_input': tf.constant([[0]], tf.int32),
    'location_input': tf.constant([0], tf.int32),
    'cmdprev_input': tf.constant([[0]], tf.int32),
    'ents2id': {".": 0},
    'entvocab_input': tf.constant([[0]], tf.int32)},
    training=True)

optim = tf.optimizers.Nadam(
    learning_rate=0.003,
    clipnorm=45.0,
    beta_1=0.9,
    beta_2=0.98)

tstamp = math.trunc(100 * time.time())

# load model if existing
modelfiles = glob.glob("{}{}*.h5".format(cwd, model_dir))
if len(modelfiles) > 0:
    fn = max(modelfiles)
    print("Loading weights:", fn)
    network.load_weights(fn)
else:
    network.save_weights('{}trained_models/{}.h5'.format(cwd, tstamp))


def get_batch(x, i, batch_size):
    return x[(i*batch_size):((i+1)*batch_size)]


def train(model, optim, data_batch):
    batch_size = len(data_batch)
    
    inputs_batch = [d['inputs'] for d in data_batch]
    cmdlist_batch = [d['cmdlist'] for d in data_batch]
    value_batch = [d['value'] for d in data_batch]
    counts_batch = [d['counts'] for d in data_batch]
    policy_batch = [np.array(x) / sum(x) for x in counts_batch]
    policy_batch = [0.98 * p + 0.02 / len(p) for p in policy_batch]

    
    # nwoutput_batch = [d['nwoutput'] for d in data_batch]  # buggy

    value_loss, policy_loss, cmdgen_loss, reg_loss = 0, 0, 0, 0
    with tf.GradientTape() as tape:
        for i in range(batch_size):
            x = inputs_batch[i]
            value = value_batch[i]
            policy = policy_batch[i]
            cmds = cmdlist_batch[i]

            cmdlist_input = tf.constant(x['cmdlist_input'], tf.int32)
            memory_input = tf.constant(x['memory_input'], tf.int32)
            cmdprev_input = tf.constant(x['cmdprev_input'], tf.int32)
            entvocab_input = tf.constant(x['entvocab_input'], tf.int32)
            location_input = tf.constant(x['location_input'], tf.int32)
            # skip round if there's only one command
            # if len(cmdlist) < 2:
            #     continue
            # evaluate model
            inputs = {'memory_input': memory_input,
                      'cmdlist_input': cmdlist_input,
                      'entvocab_input': entvocab_input,
                      'cmdprev_input': cmdprev_input,
                      'location_input': location_input,
                      'ents2id': x['ents2id']}
            output = model(inputs, training=True)
            # value loss
            vhat = output['value']
            value_loss += tf.square(value - vhat)
            # policy loss
            plogits = output['policy_logits']
            phat = tf.math.softmax(plogits)
            logphat = tf.math.log(phat + 1e-12)
            policy_loss += - tf.reduce_sum(logphat * policy)
            # cmd generation loss
            # cmdvocab = output['cmdvocab']
            # nwtoks = output['nextword_tokens']
            # nwlogits = output['nextword_logits']
            # for t, l in zip(nwtoks, nwlogits):
            #     t = tf.one_hot(t, depth=len(cmdvocab))
            #     p = tf.math.softmax(l, axis=1)
            #     logp = tf.math.log(p + 1e-12)
            #     cmdgen_loss += - tf.reduce_sum(logp * t) / len(cmdlist)
            # find next entities
            ents2id = x['ents2id']
            pad, stend = ents2id['<PAD>'], ents2id['</S>']
            nwlogits = output['nextword_logits']
            C, V, K = len(cmds), len(ents2id), nwlogits.shape[0]
            nwoutput = []
            cmdents = [[i for i in z if i != pad] for z in x['cmdprev_input']]
            j = 0
            for i in range(K - 1):
                if len(cmdents[i + 1]) > len(cmdents[i]):
                    j += 1
                    nwoutput.append(cmdents[i+1][j])
                else:
                    j = 0
                    nwoutput.append(stend)
            nwoutput.append(stend)
            # regularization loss

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
num_choice = 150
num_consider = 150
all_batchfiles = glob.glob("data/*.json")
all_batchfiles.sort(reverse=True)
all_batchfiles = all_batchfiles[:num_consider]  # exclude current

if len(all_batchfiles) > num_choice:
    datatstamps = np.random.choice(
        all_batchfiles,
        size=num_choice,
        replace=False)

# extend current data
data = []
for datafile in all_batchfiles:
    # datafile = "data/{}.json".format(s)
    print("Adding replay data from:", datafile)
    with open(datafile, 'r') as fn:
        d = ujson.load(fn)
        data.extend(d)

# data = data_current

# do this only for one big round
data = [x for x in data if
        x['value'] > 0.25 or
        x['value'] < 0.25]

# order data and obtain value policy and nextwords
data = np.random.permutation(data)

ndata = len(data)
batch_size = int(min(len(data), 8)) if len(data) > 0 else 1
num_epochs = 5 # to compare
num_batches = ndata // batch_size
ckpt_every = 160 / batch_size
# num_epochs = 2 if num_batches < 40 else 1

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
            wfile = "trained_models/{}.h5".format(tstamp)
            print("saving trained weights to {}...".format(wfile))
            network.save_weights(wfile)

            msg = "".join(["Ckpt summary: vloss: {:.2f}, ploss: {:.2f}",
                           ", cgloss: {:.2f}, rloss: {:.2f}, loss: {:.2f}"])
            print(msg.format(mv, mp, mcg, mr, ml))
            mv, mp, mcg, mr, ml = 0, 0, 0, 0, 0

tstamp = math.trunc(100 * time.time())
wfile = "trained_models/{}.h5".format(tstamp)
print("saving trained weights to {}...".format(wfile))
network.save_weights(wfile)

print(0)