import neuralnetwork
import textutils
import tensorflow as tf
import pdb

textworld_vocab = set()
with open('../TextWorld/montecarlo/vocab.txt', 'r') as fn:
    for line in fn:
        word = line[:-1]
        textworld_vocab.add(word)

embeddings, vocab = textutils.load_embeddings(
    embeddingsdir="../../glove.6B/",
    embedding_dim=100,
    vocab=textworld_vocab)

model = neuralnetwork.AlphaTextWorldNet(embeddings, vocab)
value, policy = model("you are in the kitchen", ["go west", "open fridge"])

model.memory.append("start of game")
model.memory.append("you need food")

value, policy = model("you are in the kitchen", ["go west", "open fridge", "I am starving"])

model.summary()
model.save_weights("trained_models/init.h5")