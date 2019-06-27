import tensorflow as tf
import numpy as np
import glob
import json
import pdb
import sys

import mctsagent as mcts
import attention as nn
import math
import time
from textutils import *

# load embeddings
# tf.keras.backend.set_floatx("float16")  # this doesn't work

path = '/home/mauriciogtec/'
textworld_vocab = set()
with open(path + 'Github/TextWorld/montecarlo/vocab.txt', 'r') as fn:
    for line in fn:
        word = line[:-1]
        textworld_vocab.add(word)

embedding_dim = 100
embedding_fdim = 64
embeddings, vocab = tu.load_embeddings(
    embeddingsdir=path + "glove.6B/",
    embedding_dim=embedding_dim,  # try 50
    vocab=textworld_vocab)

# np.random.seed(1989581)

index = np.random.permutation(range(embedding_dim))[:embedding_fdim]
embeddings = embeddings[index, :]

num_games = 100
max_time = 45
max_episodes = 5
gamefiles = glob.glob("games/*.ulx")
gamefiles = [gamefiles[i] for i in
             np.random.permutation(range(len(gamefiles)))]

network = nn.AlphaTextWorldNet(embeddings, vocab)

optim = tf.optimizers.Nadam(
    learning_rate=0.00001,
    clipnorm=30.0,
    beta_1=0.9,
    beta_2=0.98)

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=network)
# manager = tf.train.CheckpointManager(ckpt, root + 'ckpts', max_to_keep=100)
# ckpt.restore(manager.latest_checkpoint)

def train(model, optim, data_batch):
    batch_size = len(data_batch)
    memory_batch = [d['memory'] for d in data_batch]
    cmdlist_batch = [d['cmdlist'] for d in data_batch]
    value_batch = [d['reward'] for d in data_batch]
    counts_batch = [d['counts'] for d in data_batch]
    policy_batch = [np.array(x) / sum(x) for x in counts_batch]

    inputs_batch = zip(memory_batch, cmdlist_batch,
                       value_batch, policy_batch)
    value_loss, policy_loss, cmdgen_loss, reg_loss = 0, 0, 0, 0
    with tf.GradientTape() as tape:
        for memory, cmdlist, value, policy in inputs_batch:
            # skip round if there's only one command
            if len(cmdlist) < 2:
                continue
            # evaluate model
            inputs = {'memory': memory, 'cmdlist': cmdlist}
            output = model(inputs, training=True)
            # value loss
            vhat = output['value']
            value_loss += tf.reduce_sum(tf.square(value - vhat))
            # policy loss
            plogits = output['policy_logits']
            phat = tf.math.softmax(plogits)
            logphat = tf.math.log(phat + 1e-12)
            policy_loss += - tf.reduce_sum(logphat * policy)
            # cmd generation loss
            cmdvocab = output['cmdvocab']
            nwtoks = output['nextword_tokens']
            nwlogits = output['nextword_logits']
            for t, l in zip(nwtoks, nwlogits):
                t = tf.one_hot(t, depth=len(cmdvocab))
                p = tf.math.softmax(l, axis=1)
                logp = tf.math.log(p + 1e-12)
                cmdgen_loss += - tf.reduce_sum(logp * t) / len(cmdlist)
            # regularization loss
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


def get_batch(x, i, batch_size):
    return x[(i*batch_size):((i+1)*batch_size)]


for g in range(num_games):
    gamefile = gamefiles[g]
    print("Opening game {}".format(gamefile))

    # rain a few round with 25 to get network started
    agent = mcts.MCTSAgent(gamefile, network, cpuct=0.3, max_steps=25)

    # 1. Play and generate data ----------------------------
    gtime, ep = 0, 0
    while ep < max_episodes and gtime < max_time:
        timer = time.time()
        envscore, num_moves, infos, reward =\
            agent.play_episode(subtrees=1, max_subtree_depth=1, verbose=True)
        gtime += time.time() - timer
        msg = "game {}, episode: {:2d}, moves: {:3d}, " +\
              "envscore: {}/{}, reward: {:.2f}, time: {:.2f}"
        print(msg.format(g, ep, num_moves, envscore,
                         infos["max_score"], reward, gtime))
        ep += 1

    data_current = agent.dump_tree()

    tstamp = math.trunc(time.time())
    datafile = "data/{}.json".format(tstamp)

    with open(datafile, 'w') as fn:
        ujson.dump(data, fn)

    agent.close()

    # Let's train with the data, for the moment only this tree,
    # TODO!: replay is necessary, current form is replay is
    # remembering games, but it would be better to remember data
    # points from several random games

    # 2. Train -----------------------------------------------

    # Pull random games from last games
    num_choice = 50
    num_consider = 10
    all_datafiles = glob.glob("data/*.json")
    datatstamps = [int(x[5:-5]) for x in all_datafiles]
    datatstamps.sort(reverse=True)
    datatstamps = datatstamps[1:num_consider]  # exclude current

    # if len(datatstamps) > num_choice:
    #     datatstamps = np.random.choice(
    #         datatstamps,
    #         size=num_choice,
    #         replace=False)

    # extend current data
    for s in datatstamps:
        datafile = "data/{}.json".format(s)
        print("Adding replay data from:", datafile)
        with open(datafile, 'r') as fn:
            d = ujson.load(fn)
            data.extend(d)

    data = data_current
    data = [x for x in data if
            sum(x['counts']) > 10 and
            len(x['cmdlist']) >= 1]

    data = np.random.permutation(data)

    ndata = len(data)
    batch_size = int(len(data), 8) if len(data) > 0 else 1
    print_every = 40 / batch_size
    num_epochs = 2
    num_batches = min(ndata // batch_size, 100)
    ckpt_every = 160 / batch_size
    num_epochs = 2 if num_batches < 40 else 1

    msg = "OPTIMIZATION: epochs: {} batches: {}  time: {}"
    print(msg.format(num_epochs, num_batches, tstamp))

    for e in range(num_epochs):
        for b in range(num_batches):
            data_batch = get_batch(data, b, batch_size)

            try:
                vloss, ploss, cgloss, rloss, loss = train(
                    memory_batch, cmdlist_batch, value_batch, policy_batch)
            except Exception as e:
                print(e)
                continue

            msg = "Optimizing... epoch: {} batch: {:2d}, " +\
                "vloss: {:.2f}, ploss: {:.2f}, " +\
                "cgloss: {:.3f}, rloss {:.2f}, loss {:.2f}"

            print(msg.format(
                e, b, vloss.numpy().item(),
                ploss.numpy().item(), cgloss.numpy().item(),
                rloss.numpy().item(), loss.numpy().item()))

        wfile = "trained_models/{}.h5".format(tstamp)
        network.save_weights(wfile)

    print(0)
