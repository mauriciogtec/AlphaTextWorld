import tensorflow as tf
import numpy as np
import glob
import json
import pdb
import sys
import gc

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

embedding_dim = 100
embedding_fdim = 64
embeddings, vocab = tu.load_embeddings(
    embeddingsdir=path + "glove.6B/",
    embedding_dim=embedding_dim,  # try 50
    vocab=textworld_vocab)

# np.random.seed(1989581)

index = np.random.permutation(range(embedding_dim))[:embedding_fdim]
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
    with open(datafiles[i], 'r') as fn:
        data_array = json.load(fn)
        for d in data_array:
            rewards_list.append(d['reward'])
            cmdlist_list.append(d['cmdlist'])
            counts_list.append(d['counts'])
            memory_list.append(d['memory'])

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
manager = tf.train.CheckpointManager(ckpt, root + 'ckpts', max_to_keep=100)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


def train(memory_batch, cmdlist_batch, value_batch, policy_batch):

    value_loss, policy_loss, cmdgen_loss = 0, 0, 0

    inputs_batch = zip(memory_batch, cmdlist_batch,
                       value_batch, policy_batch)
    batch_size = len(cmdlist_batch)

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
            # add losses
        value_loss /= batch_size
        policy_loss /= batch_size
        cmdgen_loss /= batch_size
        loss = value_loss + policy_loss + cmdgen_loss
    # apply gradients
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    update = optim.apply_gradients(zip(gradients, variables))

    gc.collect()

    return value_loss, policy_loss, cmdgen_loss, loss

ndata = len(cmdlist_list)
batch_size = 4
print_every = 40 / batch_size
num_epochs = 2
num_batches = ndata // batch_size
ckpt_every = 160 / batch_size
iteration = 0
lam = 0.05 / 8 * batch_size
vloss_av = 0.2
ploss_av = 0.8
cgloss_av = 7.4
loss_av = vloss_av + ploss_av + cgloss_av

msg = "".join(["epoch: {}, iter: {}, vloss (av): {:.2f}, ploss (av): {:.2f}, ",
              "cgloss(av): {:.2f}, loss (av): {:.2f}"])


def get_batch(x, i, batch_size):
    return x[(i*batch_size):((i+1)*batch_size)]


for epoch in range(num_epochs):
    for i in range(num_batches):

        memory_batch = get_batch(memory_list, i, batch_size)
        cmdlist_batch = get_batch(cmdlist_list, i, batch_size)
        value_batch = get_batch(rewards_list, i, batch_size)
        counts_batch = get_batch(counts_list, i, batch_size)
        policy_batch = [np.array(x) / sum(x) for x in counts_batch]

        try:
            vloss, ploss, cgloss, loss = train(
                memory_batch, cmdlist_batch, value_batch, policy_batch)
        except Exception as e:
            print(e)
            continue

        vloss_av += lam * (vloss.numpy().item() - vloss_av)
        ploss_av += lam * (ploss.numpy().item() - ploss_av)
        cgloss_av += lam * (cgloss.numpy().item() - cgloss_av)
        loss_av += lam * (loss.numpy().item() - loss_av)

        if iteration % print_every == 0:
            print(msg.format(epoch, (i + 1) * batch_size,
                                vloss_av, ploss_av, cgloss_av, loss_av))

        if iteration % ckpt_every == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(
                int(ckpt.step), save_path))
            print("loss {:1.2f}".format(loss.numpy()))

        iteration += 1
