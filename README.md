# MCTS + DL -> TextWorld

This project implements an artificial agent for cometing at the TextWorld challenge ([aka.ms/textworld](aka.ms/textworld))

The file `mctsagent.py` defines a Monte Carlo Tree Search solves TextWorld games. It uses PUCT search which uses a neural network to evaluate unseen game states.

The file `attention.py` defines the neural network that takes as input an array of observations (or memory) and learns to predict the value fo the game given those observations, a policy for the next decision, and a language model to generate commands.

