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

cwd = './'

sys.path.append(cwd)
from custom_layers import *
from textutils import *
import attention as nn
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

s = ""
for word in vocab:
    s += word + "\n"

with open('final_vocab.txt', 'w') as fn:
    fn.write(s)