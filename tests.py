import neuralnetwork
import textutils
import tensorflow as tf
import pdb
import mcmcagent
import pickle

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
initrun = model("you are in the ", ["."])
model.load_weights("trained_models/init.h5")

gamefile = "../TextWorld/notebooks/training_games/tw-simple-rDense+gDetailed+train-house-GP-0a72tMGVtQnPSvMR.ulx"

agent = mcmcagent.MCTSAgent(gamefile, model, cpuct=0.1)

num_games = 100
msg = "finished game {:2d} in {:3d} moves with score {:d}/{:d}"
for i in range(num_games):
    score, num_moves, infos = agent.play_game()
    print(msg.format(i + 1, num_moves, score, infos["max_score"]))

print("Root --------")
print(agent.root)
print("Root Edges --")
for edge in agent.root.edges:
    print(edge)

data = agent.dump_data()

with open('data100_cpuct_0_1.pkl', 'wb') as fn:
    pickle.dump(data, fn)
print(0)
