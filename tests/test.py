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
    embedding_dim=300,
    vocab=textworld_vocab)

model = neuralnetwork.AlphaTextWorldNet(embeddings, vocab)

model.add_to_memory("start of game")

obsx, cmdlistx = model("you need food", ["go west", "open fridge"])
