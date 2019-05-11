import neuralnetwork as nn
import textutils as tu
import numpy as np
import tensorflow as tf
import mctsagent as mcts
import glob
import time
import ujson
import math

import pdb

textworld_vocab = set()
with open('../TextWorld/montecarlo/vocab.txt', 'r') as fn:
    for line in fn:
        word = line[:-1]
        textworld_vocab.add(word)

embeddings, vocab = tu.load_embeddings(
    embeddingsdir="../../glove.6B/",
    embedding_dim=100,
    vocab=textworld_vocab)


# network = nn.AlphaTextWorldNet(embeddings, vocab)
# testrun = network(
#     "you are in thehi ther ",
#     [".", ".."],
#     memory=[".", "."],
#     training=True)
# network.save_weights("trained_models/init.h5")

# print(network.memory)
# network.flush_memory()

# network.play_mode()
# testrun = network("you are in the ", ["."])
# network.save_weights("trained_models/init.h5")
# print(network.memory)
# network.flush_memory()

# network.load_weights("trained_models/init.h5")

num_games = 100
max_time = 180
max_episodes = 3
gamefiles = glob.glob("games/*.ulx")
gamefiles = [gamefiles[i] for i in
             np.random.permutation(range(len(gamefiles)))]

msg = "game {}, episode: {:2d}, moves: {:3d}, \
       score: {}/{}, reward: {:.2f}, time: {:.2f}"

network = nn.load_network(
    embeddings, vocab,
    "trained_models/init_pretrained.h5")

optim = tf.optimizers.Nadam(
    learning_rate=0.0005,
    clipnorm=30.0,
    beta_1=0.5,
    beta_2=0.75)

for g in range(num_games):
    gamefile = gamefiles[g]
    # rain a few round with 25 to get network started
    agent = mcts.MCTSAgent(gamefile, network, cpuct=0.1, max_steps=50)

    # 1. Play and generate data ----------------------------
    gtime, ep = 0, 0
    while ep < max_episodes and gtime < max_time:
        timer = time.time()
        score, num_moves, infos, reward =\
            agent.play_episode(subtrees=8, max_subtree_depth=8, verbose=True)
        gtime += time.time() - timer
        print(msg.format(g, ep, num_moves, score,
                         infos["max_score"], reward, gtime))

        if num_moves > 10:
            ep += 1

    print("Root --------")
    print(agent.root)
    print("Root Edges --")
    for edge in agent.root.edges:
        print(edge)

    data = agent.dump_tree()
    datafile = "data/tree_{}.json".format(gamefile[6:-4])

    with open(datafile, 'w') as fn:
        ujson.dump(data, fn)

    agent.close()

    # Let's train with the data, for the moment only this tree,
    # TODO!: replay is necessary

    # 2. Train -----------------------------------------------
    # datafile = '/home/mauriciogtec/Github/AlphaTextWorld/data/tree_tw-simple-rDense+gDetailed+train-house-GP-p1RLFX7MU7KjFy0d.json'
    with open(datafile, 'r') as fn:
        data = ujson.load(fn)

    batch_size = 16  # larger means more memory
    epochs = 1
    data = np.random.permutation(data)
    num_batches = len(data) // batch_size

    print("Optimizing {} batches per epoch".format(num_batches))

    for e in range(epochs):
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
                    probs = np.array(datum['counts'], dtype=np.float32) + 1e-3
                    probs /= probs.sum()
                    eps = 0.1 / K
                    dnoise = np.random.dirichlet(np.ones(K))
                    dnoise = dnoise.astype(np.float32)
                    probs = (1.0 - eps) * probs + eps * dnoise

                    # Policy loss
                    reward = datum['reward']
                    vloss += 0.5 * tf.reduce_sum(
                        tf.square(vhat - reward)) / batch_size
                    logphat = tf.math.log(phat + 1e-6)
                    ploss += - tf.reduce_sum(logphat * probs) / batch_size
                    rloss += tf.math.add_n(
                        [l for l in network.losses
                            if not np.isnan(l.numpy())]) / batch_size
                loss = vloss + ploss + rloss

                if np.isnan(loss.numpy().item()):
                    pdb.set_trace()

            msg = "Optimizing... epoch: {} batch: {:3d}, " +\
                  "loss: {:.2f}, ploss: {:.2f}, " +\
                  "rloss: {:.2f}, vloss {:.2f}"
            print(msg.format(
                e, b,
                loss.numpy().item(), ploss.numpy().item(),
                rloss.numpy().item(), vloss.numpy().item()))
            gradients = tape.gradient(loss, network.trainable_variables)
            update = optim.apply_gradients(
                zip(gradients, network.trainable_variables))

    wfile = "trained_models/w{:05d}.h5".format(math.trunc(time.time()))
    network.save_weights(wfile)

    # data = agent.dump_data()

    # with open('data100_cpuct_0_1.pkl', 'wb') as fn:
    #     pickle.dump(data, fn)
    print(0)
