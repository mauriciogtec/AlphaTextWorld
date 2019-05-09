import tensorflow as tf
import numpy as np
from typing import List
import re


def get_word_id(word: str, word2id: dict):
    key = word if word in word2id else '<UNK>'
    return word2id[key]


def tokenize(text: str, word2id: dict):
    text = re.sub("'", "", text)
    text = re.sub("[.]+", ".", text)
    text = re.sub("[^a-zA-Z0-9\-\., ]", r" ", text)
    text = re.sub("([,\.!])", r" \1 ", text)
    word_ids = [get_word_id(w, word2id) for w in text.split()]
    return word_ids


def text2tensor(texts: List[str], word2id: dict):
    toktexts = [tokenize(t, word2id) for t in texts]
    maxlen = max(len(t) for t in toktexts)
    padded = np.full((len(texts), maxlen), word2id["<PAD>"])
    for i, t in enumerate(toktexts):
        padded[i, :len(t)] = t
    padded = tf.constant(padded, dtype=tf.int32)
    return padded


def load_embeddings(embeddingsdir: str, embedding_dim: int, vocab: List[str]):
    """Loads embeddings only for vocab"""
    path_to_embeddings = '{}glove.6B.{:d}d.txt'.\
        format(embeddingsdir, embedding_dim)
    columns = [np.zeros(shape=(embedding_dim, 2), dtype='float32')]
    voc = ["<PAD>", "<UNK>", "<S>", "</S>"]
    voc_size = 4  # num_words + two special tokens

    with open(path_to_embeddings, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                embedding = np.array(values[1:], dtype='float32')
                embedding = embedding.reshape(-1, 1)
                voc.append(word)
                columns.append(embedding)
                voc_size += 1
    embeddings = np.hstack(columns)
    return embeddings, voc
