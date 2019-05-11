import neuralnetwork as nn
import textutils as tu
import numpy as np
import tensorflow as tf
import pdb
import mctsagent as mcts
import pickle
import glob
import time
import ujson
import importlib
import gc

mcts = importlib.reload(mcts)

textworld_vocab = set()
with open('../TextWorld/montecarlo/vocab.txt', 'r') as fn:
    for line in fn:
        word = line[:-1]
        textworld_vocab.add(word)

embeddings, vocab = tu.load_embeddings(
    embeddingsdir="../../glove.6B/",
    embedding_dim=100,
    vocab=textworld_vocab)


# model = nn.AlphaTextWorldNet(embeddings, vocab)
# model.memory.append(".")
# model.tensor_memory = False
# testrun = model("you are in thehi ther ", [".", ".."], training=True)
# testrun = model("you are in thehi ther ", [".", "..", "..."], training=True)
# model.save_weights("trained_models/init.h5")
# model = []

# print(model.memory)
# model.flush_memory()

# model.play_mode()
# testrun = model("you are in the ", ["."])
# model.save_weights("trained_models/init.h5")
# print(model.memory)
# model.flush_memory()

# model.load_weights("trained_models/init.h5")

num_games = 100
max_time = 360
max_episodes = 5
gamefiles = glob.glob("games/*.ulx")
gamefiles = [gamefiles[i] for i in
             np.random.permutation(range(len(gamefiles)))]

msg = "game {}, episode: {:2d}, moves: {:3d}, \
       score: {}/{}, reward: {:.2f}, time: {:.2f}"

network = nn.load_network(embeddings, vocab, "trained_models/init.h5")
optim = tf.optimizers.Nadam(
    learning_rate=0.001,
    clipnorm=30.0,
    beta_1=0.5,
    beta_2=0.75)

for g in range(num_games):
    gamefile = gamefiles[g]
    # rain a few round with 25 to get network started
    agent = mcts.MCTSAgent(gamefile, network, cpuct=2.4, max_steps=25)
    
    # agent.network.tensor_memory = True  # faster

    # 1. Play and generate data ----------------------------
    gtime, ep = 0, 0
    while ep < max_episodes and gtime < max_time:
        timer = time.time()
        score, num_moves, infos, reward =\
            agent.play_episode(subtrees=200, verbose=True)
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
    agent = []
    gc.collect()

    # Let's train with the data, for the moment only this tree,
    # TODO!: replay is necessary

    # 2. Train -----------------------------------------------
    model = agent.network
    # model.tensor_memory = False  # False | needed for training

    batch_size = 16
    epochs = 5
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
                    model.memory = datum['memory']
                    obs, cmdlist = datum['obs'], datum['cmdlist']
                    vhat, phat = model(obs, cmdlist, training=True)

                    # create target prob vector
                    probs = np.array(datum['counts'])
                    probs /= probs.sum()
                    eps = 0.1 * (1 / len(cmdlist))
                    dnoise = np.random.dirichlet(np.ones(len(cmdlist)))
                    probs = (1.0 - eps) * probs + eps * dnoise

                    # Policy loss
                    value = datum['reward']
                    vloss += 0.5 * tf.reduce_sum(
                        tf.square(vhat - value)) / batch_size
                    logphat = tf.math.log(phat + 1e-6)
                    ploss += - tf.reduce_sum(logphat * probs) / batch_size
                    rloss += tf.math.add_n(
                        [l for l in model.losses
                         if not np.isnan(l.numpy())]) / batch_size
                loss = vloss + ploss + rloss

                if np.isnan(loss.numpy().item()):
                    pdb.set_trace()

            msg = "Optimizing... loss: {:.2f}, ploss: {:.2f}, " +\
                  "rloss: {:.2f}, vloss {:.2f}"
            print(msg.format(loss.numpy().item(), ploss.numpy().item(),
                             rloss.numpy().item(), vloss.numpy().item()))
            gradients = tape.gradient(loss, model.trainable_variables)
            optim.apply_gradients(zip(gradients, model.trainable_variables))

    wfile = "trained_models/w{:05d}.h5".format(g)
    model.save_weights(wfile)

# data = agent.dump_data()

# with open('data100_cpuct_0_1.pkl', 'wb') as fn:
#     pickle.dump(data, fn)
print(0)
