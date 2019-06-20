import tensorflow as tf
import numpy as np
import glob
import json
import pdb
import sys

sys.path.append("../")
root = "/home/mauriciogtec/Github/AlphaTextWorld/"
sys.path.append(root)
import textutils as tu
import attention as nn

# load embeddings
# tf.keras.backend.set_floatx("float16")  # this doesn't work

path = '/home/mauriciogtec/'
textworld_vocab = set()
with open(path + 'Github/TextWorld/montecarlo/vocab.txt', 'r') as fn:
    for line in fn:
        word = line[:-1]
        textworld_vocab.add(word)

embeddings, vocab = tu.load_embeddings(
    embeddingsdir=path + "glove.6B/",
    embedding_dim=200,  # try 50
    vocab=textworld_vocab)
np.random.seed(110104)
index = np.random.permutation(range(200))[:128]
embeddings = embeddings[index, :]


# load data files
datafiles = glob.glob(root + "data/*.json")
datafiles.sort(reverse=True)

num_files = 50
cmdlist_list = []
memory_list = []
counts_list = []
rewards_list = []
for i in range(num_files):
    with open(datafiles[i],'r') as fn:
        data_array = json.load(fn)
        for d in data_array:
            rewards_list.append(d['reward'])
            cmdlist_list.append(d['cmdlist'])
            counts_list.append(d['counts'])
            memory_list.append(d['memory'])

tu.noun_phrases(memory_list[1000])

N = len(cmdlist_list)
idx = np.random.permutation(range(N))
cmdlist_list = [cmdlist_list[i] for i in idx]
memory_list = [memory_list[i] for i in idx]
counts_list = [counts_list[i] for i in idx]
rewards_list = [rewards_list[i] for i in idx]

print("number data points")
print(len(cmdlist_list))


model = nn.AlphaTextWorldNet(embeddings, vocab)

optim = tf.optimizers.Nadam(
    learning_rate=0.00001,
    clipnorm=30.0,
    beta_1=0.9,
    beta_2=0.98)


ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=model)
manager = tf.train.CheckpointManager(ckpt, root + 'ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


def train(memory, cmdlist, value, policy):
    inputs = (memory, cmdlist)
    value_loss, policy_loss, loss = 0, 0, 0
    with tf.GradientTape() as tape:
        vhat, phat = model(inputs, training=True)
        value_loss += tf.math.reduce_sum(tf.square(value - vhat))
        phat = tf.math.softmax(phat)
        logphat = tf.math.log(phat + 1e-12)
        policy_loss += -tf.reduce_sum(logphat * policy)
        loss += value_loss + policy_loss

    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    update = optim.apply_gradients(zip(gradients, variables))
    
    return value_loss, policy_loss, loss

# out = model((["this", "is a test"], ["hi there", "how are you here"]), training=True)
# model.summary()

print_every = 10
ckpt_every = 100
iteration = 0
vloss_av = 1.0
ploss_av = 1.5
loss_av = 2.5

msg = "epoch: {}, iter: {}, vloss: {:.2f}, ploss: {:.2f}, loss: {:.2f}, " +\
      "vloss (av): {:.2f}, ploss (av): {:.2f}, loss (av): {:.2f}"

with tf.device('/gpu:0'):
    for epoch in range(5):
        for i in range(2000, len(cmdlist_list)):

            cmdlist = cmdlist_list[i]
            memory = memory_list[i]
            value = rewards_list[i]
            counts = np.array(counts_list[i])
            policy = counts / sum(counts)
            maxlen = max(len(m) for m in memory)

            if len(cmdlist) >= 2 and len(memory) > 0:
                vloss, ploss, loss = train(memory, cmdlist, value, policy)

                vloss_av += 0.01 * (vloss.numpy().item() - vloss_av)
                ploss_av += 0.01 * (ploss.numpy().item() - ploss_av)
                loss_av += 0.01 * (loss.numpy().item() - loss_av)

                if iteration % print_every == 0:
                    print(msg.format(epoch, iteration,
                                     vloss, ploss, loss,
                                     vloss_av, ploss_av, loss_av))

                if iteration % ckpt_every == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(
                        int(ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss.numpy()))

                iteration += 1
