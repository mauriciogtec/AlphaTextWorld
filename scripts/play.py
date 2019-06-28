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
description = "Play a round of games."
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--gameindex',
                    type=int, default=1,
                    help='Number of games to be played.')
parser.add_argument('--subtrees',
                    type=int, default=50,
                    help='Subtrees to spawn.')
parser.add_argument('--subtree_depth',
                    type=int, default=2,
                    help='Max depth of search trees.')
parser.add_argument('--max_steps',
                    type=int, default=100,
                    help='Max number of steps per game. Defaults to 100.')
parser.add_argument('--min_time',
                    type=float, default=10,
                    help=''.join(['Min time playing. If a game ends sooner',
                                  ', it will play another episode.']))
parser.add_argument('--cwd', default='.',
                    help='Directory from which to launch')
parser.add_argument('--output_dir', default='data/',
                    help='Directory in which to save game results')
args = parser.parse_args()
print("Arguments: ", args)

cwd = args.cwd
gameindex = args.gameindex
min_time = args.min_time
max_steps = args.max_steps
subtrees = args.subtrees
subtree_depth = args.subtree_depth
output_dir = args.output_dir

sys.path.append(cwd)
from custom_layers import *
from textutils import *
import attention as nn
import mctsagent as mcts
# ----------------------

textworld_vocab = set()
with open(cwd + '/textworld_vocab.txt', 'r') as fn:
    for line in fn:
        word = line[:-1]
        textworld_vocab.add(word)

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


network = nn.AlphaTextWorldNet(embeddings, vocab)
network(inputs={
    'memory_input': tf.constant([[0]], tf.int32),
    'cmdlist_input': tf.constant([[0]], tf.int32),
    'location_input': tf.constant([0], tf.int32),
    'cmdprev_input': tf.constant([[0]], tf.int32),
    'ents2id': {".": 0},
    'entvocab_input': tf.constant([[0]], tf.int32)},
    training=True)

# load latest weights if available
modeldir = cwd + "trained_models/"
models = glob.glob(modeldir + "*.h5")
if len(models) > 0:
    latest = max(models)
    network.load_weights(latest)


# rain a few round with 25 to get network started
gamefiles = glob.glob(cwd + "/games/*.ulx")
# gamefile = gamefiles[gameindex]
gamefile = np.random.choice(gamefiles)
print("Opening game {}".format(gamefile))
agent = mcts.MCTSAgent(
    gamefile, network,
    cpuct=0.4,
    dnoise=0.03,
    max_steps=max_steps,
    temperature=0.3)

# Play and generate data ----------------------------
t = 0.0
data = []
while t < min_time:
    timer = time.time()
    envscore, num_moves, infos, reward = agent.play_episode(
        subtrees=subtrees,
        max_subtree_depth=subtree_depth,
        verbose=True)
    msg = "moves: {:3d}, envscore: {}/{}, reward: {:.2f}"
    print(msg.format(num_moves, envscore, infos["max_score"], reward))
    data.extend(agent.dump_tree(mainbranch=True))
    t += time.time() - timer

tstamp = math.trunc(100 * time.time())
# datafile = cwd + "/{}/{}.json".format(output_dir, tstamp)
# with open(datafile, 'w') as fn:
#     ujson.dump(data, fn)
print(0)