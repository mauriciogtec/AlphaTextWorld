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
import h5py

# import socket
# from time import sleep

# def work(jobnum):
#     print("Starting job on {}.".format(socket.gethostname()))
#     # print("Finished job {}...\n".format(jobnum))
# work(0)

# ----------------------
description = "Play a round of games."
parser = argparse.ArgumentParser(description=description)
# parser.add_argument('--gameindex',
#                     type=int, default=1,
#                     help='Number of games to be played.')
parser.add_argument('--num_games',
                    type=int, default=10,
                    help='Number of games to play.')
parser.add_argument('--cpuct',
                    type=float, default=0.4,
                    help='MCTS Exploration UCT constant')
parser.add_argument('--subtrees',
                    type=int, default=50,
                    help='Subtrees to spawn.')
parser.add_argument('--temperature',
                    type=float, default=0.5,
                    help='determines randomness')
parser.add_argument('--subtree_depth',
                    type=int, default=5,
                    help='Max depth of search trees.')
parser.add_argument('--max_steps',
                    type=int, default=25,
                    help='Max number of steps per game. Defaults to 100.')
parser.add_argument('--min_time',
                    type=float, default=15,
                    help=''.join(['Min time playing. If a game ends sooner',
                                  ', it will play another episode.']))
parser.add_argument('--verbose',
                    type=bool, default=False,
                    help='Prints every game')
parser.add_argument('--cwd', default='./',
                    help='Directory from which to launch')
parser.add_argument('--output_dir', default='data/',
                    help='Directory in which to save game results')
args = parser.parse_args()
# print("Arguments: ", args)

cwd = args.cwd
num_games = args.num_games
# gameindex = args.gameindex
min_time = args.min_time
max_steps = args.max_steps
subtrees = args.subtrees
subtree_depth = args.subtree_depth
output_dir = args.output_dir
temperature = args.temperature
verbose = args.verbose
cpuct = args.cpuct

sys.path.append(cwd)
from custom_layers import *
from textutils import *
import attention2 as nn
import mctsagent as mcts
# ----------------------

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

# load latest weights if available
modeldir = cwd + "trained_models2/"
models = glob.glob(modeldir + "*.h5")
if len(models) > 0:
    latest = max(models)
    network.load_weights(latest)

# rain a few round with 25 to get network started
gamefiles = glob.glob(cwd + "../allgames/*.ulx")

# gamefile = gamefiles[gameindex]
# np.random.seed(time.time())

for _ in range(num_games):
    i = np.random.choice(len(gamefiles))
    gamefile = gamefiles[i]
    # print(gamefiles)
    # gamefile = gamefiles[2]
    # print(gamefile)
    # if verbose:
    #     print("Opening game {}".format(gamefile))

    agent = mcts.MCTSAgent(
        gamefile, network,
        cpuct=cpuct,
        dnoise=0.05,
        max_steps=max_steps,
        temperature=temperature)
    agent.gamefile

    # Play and generate data ----------------------------
    t = 0.0
    data = []
    while t < min_time:
        timer = time.time()
        envscore, num_moves, infos, reward = agent.play_episode(
            subtrees=subtrees,
            max_subtree_depth=subtree_depth,
            verbose=verbose)
        msg = "moves: {:3d}, envscore: {}/{}, reward: {:.2f}"
        # if verbose:
        #     print(msg.format(num_moves, envscore, infos["max_score"], reward))
        data.extend(agent.dump_tree(mainbranch=True))
        t += time.time() - timer

    tstamp = math.trunc(100 * time.time())
    datafile = cwd + "{}/{}.json".format(output_dir, tstamp)
    with open(datafile, 'w') as fn:
        ujson.dump(data, fn)
print(0)