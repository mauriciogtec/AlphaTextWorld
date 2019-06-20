import tensorflow as tf
import numpy as np
from typing import List
import nltk
import re
from collections import deque


def get_word_id(word: str, word2id: dict):
    key = word if word in word2id else '<UNK>'
    return word2id[key]


def clean_text(text: str):
    text = text.lower()
    text = re.sub("'", "", text)
    text = re.sub("[.]+", ".", text)
    text = re.sub("[^a-zA-Z0-9\-\.,\?\! ]", r" ", text)
    text = re.sub("[ ]+", " ", text)
    text = re.sub("([,\.!])", r" \1 ", text)
    return text


def tokenize(text: str, word2id: dict, clean: bool = True):
    if clean:
        text = clean_text(text)
    word_ids = [get_word_id(w, word2id) for w in text.split()]
    return word_ids


def tokenize_list(texts: List[str], word2id: dict,
                  max_token_length: int=64, rewind: int=8):
    ans_ids = []
    ans_words = []
    clean_texts = []
    for t in texts:
        t = clean_text(t)
        ts = re.split("\.", t)
        ts = [x.strip() for x in ts]
        ts = [x for x in ts if x != ""]
        if len(ts) == 1:
            clean_texts.append(t)
        else:
            for i in range(len(ts) - 1):
                t = "{} . {}".format(ts[i], ts[i + 1])
                clean_texts.append(t)

    buffer = deque(clean_texts)
    text = buffer.popleft()
    thissplit = deque(text.split())

    while True:
        current_ids = []
        current_words = []
        while len(current_ids) < max_token_length and len(thissplit) > 0:
            nextword = thissplit.popleft()
            nextid = get_word_id(nextword, word2id)
            current_ids.append(nextid)
            current_words.append(nextword)

        ans_ids.append(current_ids)
        ans_words.append(current_words)

        if len(thissplit) == 0:  # we are done here
            if len(buffer) > 0:
                text = buffer.popleft()
                thissplit = deque(text.split())
            else:
                break
        else:
            thissplit.extendleft(current_words[-rewind:])

    return ans_ids, ans_words


def text2tensor(texts: List[str], word2id: dict, max_token_length: int=64):
    toktexts, _ = tokenize_list(texts, word2id)
    maxlen = max(len(t) for t in toktexts)
    padded = np.full((len(toktexts), maxlen), word2id["<PAD>"])
    for i, t in enumerate(toktexts):
        padded[i, :len(t)] = t
    padded = tf.constant(padded, dtype=tf.int32)
    return padded


def text_to_tensor_list(texts: List[str], word2id: dict):
    toktexts, _ = tokenize_list(texts, word2id)
    return [tf.constant(x, tf.int32) for x in toktexts]


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


def noun_phrases(texts):
    document = ""
    for t in texts:
        document += ". " + clean_text(t)

    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    grammar = "NP: {<DT>?<JJ>*<NN>+}"
    cp = nltk.RegexpParser(grammar)
    noun_phrases = []
    for s in sentences:
        result = cp.parse(s)
        for x in result.subtrees():
            if x.label() == 'NP':
                wlist = [x[0] for x in x]
                N = len(wlist)
                for i in range(1, N):
                    buffer = wlist[i]
                    for j in range(i + 1, N):
                        buffer += ' ' + wlist[j]
                    noun_phrases.append(buffer)
    return noun_phrases
