import neuralnetwork as nn
import textutils as tu
import numpy as np
import tensorflow as tf
import mctsagent as mcts
import glob
import time
import ujson
import math
import re

import pdb

textworld_vocab = set()
with open('../TextWorld/montecarlo/vocab.txt', 'r') as fn:
    for line in fn:
        word = line[:-1]
        textworld_vocab.add(word)

embeddings, vocab = tu.load_embeddings(
    embeddingsdir="../../glove.6B/",
    embedding_dim=100,  # try 50
    vocab=textworld_vocab)


# network = nn.AlphaTextWorldNet(embeddings, vocab)
# testrun = network(
#     "you are in thehi ther ",
#     [".", ".."],
#     memory=[".", "."],
#     training=True)
# network.save_weights("trained_models/init.h5")


num_games = 100
max_time = 45
max_episodes = 5
gamefiles = glob.glob("games/*.ulx")
gamefiles = [gamefiles[i] for i in
             np.random.permutation(range(len(gamefiles)))]

network = nn.load_network(
    embeddings, vocab,
    "trained_models/1557898363.h5")

optim = tf.optimizers.Nadam(
    learning_rate=0.00003,
    clipnorm=30.0,
    beta_1=0.7,
    beta_2=0.9)

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
            agent.play_episode(subtrees=50, max_subtree_depth=16, verbose=True)
        gtime += time.time() - timer
        msg = "game {}, episode: {:2d}, moves: {:3d}, " +\
              "envscore: {}/{}, reward: {:.2f}, time: {:.2f}"
        print(msg.format(g, ep, num_moves, envscore,
                         infos["max_score"], reward, gtime))
        ep += 1

    data = agent.dump_tree()

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
    num_choice = 10
    num_consider = 25
    all_datafiles = glob.glob("data/*.json")
    datatstamps = [int(x[5:-5]) for x in all_datafiles]
    datatstamps.sort(reverse=True)
    datatstamps = datatstamps[1:num_consider]  # exclude current

    if len(datatstamps) > num_choice:
        datatstamps = np.random.choice(
            datatstamps,
            size=num_choice,
            replace=False)

    # extend current data
    for s in datatstamps:
        datafile = "data/{}.json".format(s)
        print("Adding replay data from:", datafile)
        with open(datafile, 'r') as fn:
            d = ujson.load(fn)
            data.extend(d)

    data = [x for x in data if
            sum(x['counts']) > 10 and
            len(x['cmdlist']) > 2]  # ensure quality data

    data = np.random.permutation(data)

    batch_size = min(len(data), 8) if len(data) > 0 else 1
    num_batches = min(len(data) // batch_size, 100)

    num_epochs = 2 if num_batches < 40 else 1

    msg = "OPTIMIZATION: epochs: {} batches: {}  time: {}"
    print(msg.format(num_epochs, num_batches, tstamp))

    for e in range(num_epochs):
        for b in range(num_batches):
            batch = data[(b * batch_size):((b + 1) * batch_size)]
            loss = 0
            vloss, ploss, rloss = 0, 0, 0

            with tf.GradientTape() as tape:
                for datum in batch:
                    K = len(datum['cmdlist'])
                    vhat, phat = network(
                        datum['obs'],
                        datum['cmdlist'],
                        memory=datum['memory'],
                        training=True)

                    # Create target prob vector
                    probs = np.array(datum['counts'], dtype=np.float32)
                    probs /= probs.sum()
                    eps = 0.1
                    dnoise = np.random.dirichlet(np.ones(K))
                    dnoise = dnoise.astype(np.float32)
                    probs = (1.0 - eps) * probs + eps * dnoise

                    # Policy loss
                    reward = datum['reward']
                    logphat = tf.math.log(phat + 1e-6)
                    vloss += 10 * tf.losses.Huber(delta=0.5)(
                        vhat, reward) / batch_size
                    ploss += - tf.reduce_sum(logphat * probs) / batch_size
                    rloss += tf.math.add_n(
                        [l for l in network.losses
                         if not np.isnan(l.numpy())]) / batch_size
                loss = vloss + ploss + rloss

            msg = "Optimizing... epoch: {} batch: {:2d}, " +\
                  "loss: {:.2f}, ploss: {:.2f}, " +\
                  "rloss: {:.3f}, vloss {:.2f}"
            print(msg.format(
                e, b,
                loss.numpy().item(), ploss.numpy().item(),
                rloss.numpy().item(), vloss.numpy().item()))
            gradients = tape.gradient(loss, network.trainable_variables)
            update = optim.apply_gradients(
                zip(gradients, network.trainable_variables))

            wfile = "trained_models/{}.h5".format(tstamp)
            network.save_weights(wfile)

    # data = agent.dump_data()

    # with open('data100_cpuct_0_1.pkl', 'wb') as fn:
    #     pickle.dump(data, fn)
    print(0)
