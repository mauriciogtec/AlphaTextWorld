import tensorflow as tf
import numpy as np
from typing import List
import nltk
import re
from collections import deque

USELESS_FEEDBACK = [
    "That's not a verb I recognise",
    "You can't see any such thing",
    "You can't go that way"]
VALID_DIRECTIONS = ["north", "south", "west", "east"]
UNWANTED_WORDS = ['a', 'an', 'the']


def get_word_id(word: str, word2id: dict):
    key = word if word in word2id else '<UNK>'
    return word2id[key]


def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"'", "", text)
    # text = re.sub(r"\n[\n]+", " . ", text)
    text = re.sub(r"\n", " . ", text)
    text = re.sub(r"[^a-z0-9\-\.,\?\!= ]", " ", text)
    text = re.sub(r"([^a-z])", r" \1 ", text)
    text = re.sub(r"[ ]+", " ", text)
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
        ts = t.split(".")
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


def text2tensor(texts: List[str], word2id: dict, max_token_length: int=None):
    if max_token_length:
        toktexts, _ = tokenize_list(texts, word2id)
    else:
        toktexts = [[get_word_id(w, word2id) for w in s.split()]
                    for s in texts]
    maxlen = max([len(t) for t in toktexts])
    pad = word2id["<PAD>"]
    rows = [t + [pad] * (maxlen - len(t)) for t in toktexts]
    padded = np.array(rows, dtype=int)
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
                wlist = [z[0] for z in x if z[1] != "DT"]
                N = len(wlist)
                for i in range(1, N):
                    buffer = wlist[i]
                    for j in range(i + 1, N):
                        buffer += ' ' + wlist[j]
                    noun_phrases.append(buffer)
    noun_phrases = list(set(noun_phrases))
    return noun_phrases


# def get_locations_and_directions(self, memory):
#     locs = []
#     idxs = []
#     dirs = []
#     valid_dirs = ["west", "east", "north", "south"]
#     for i, txt in enumerate(memory):
#         search = re.search("= ([A-Za-z]+) =", txt)
#         if search:
#             l = search.group(1).lower()
#             words = txt.split()
#             d = [d for d in valid_dirs if d in words]
#             locs.append(l)
#             dirs.append(d)
#             idxs.apend(i)
#     return locs, dirs, idxs


def paste_collapse(strings, sep=' '):
    s = ''
    if len(strings) > 0:
        s = strings[0]
        for w in strings[1:]:
            s += sep + w
    return s

def tokenize_from_cmd_template(cmd):
    words = [x for x in cmd.split() if x not in UNWANTED_WORDS]
    template = [words[0]]
    i = 1
    s = words[1]
    while i < len(words) - 1:
        if words[i + 1] not in ADVERBS:
            s += ' ' + words[i + 1]
            i += 1
        else:
            template.append(s)
            template.append(words[i + 1])
            s = words[i + 2]
            i += 2
    template.append(s)
    return template


class FeedbackMeta:
    def __init__(self, text):
        self.text = text
        self.is_valid = None
        self.clean_text = None
        self.sentences = None
        self.location = None
        self.is_location = None
        self.directions = None
        self.entities = None

        self._identify_valid()
        if self.is_valid:
            self._preprocess_text()
            self._identify_location_and_directions()
            self._identify_entities()

    def _identify_valid(self):
        is_valid = []
        self.is_valid = self.text not in USELESS_FEEDBACK

    def _preprocess_text(self):
        x = clean_text(self.text)
        x = re.sub("= [a-z]+ =", "", x)
        sents = nltk.sent_tokenize(x)
        expr = "^[^a-z]+|[^a-z]+$"
        sents = [re.sub(expr, "", s) for s in sents]
        sents = [s for s in sents if s != '']
        if len(sents) == 0:
            self.sentences = []
            self.clean_text = ""
        else:
            self.sentences = sents
            cl = sents[0]
            for i in range(1, len(sents)):
                cl += " . " + sents[i]
            self.clean_text = cl

    def _identify_location_and_directions(self):
        if self.is_valid:
            search = re.search("= ([\w]+) =", self.text)
            if search:
                self.is_location = True
                self.location = search.group(1).lower()
                words = set(self.clean_text.split())
                self.directions = []
                for d in VALID_DIRECTIONS:
                    if d in words:
                        self.directions.append(d)

    def _identify_entities(self):
        tokens = [nltk.word_tokenize(s) for s in self.sentences]
        postags = [nltk.pos_tag(s) for s in tokens]
        grammar = "NP: {<DT>?<JJ>*<NN>+}"
        cp = nltk.RegexpParser(grammar)
        entities = []
        for s in postags:
            result = cp.parse(s)
            for tree in result.subtrees():
                if tree.label() == 'NP':
                    # remove article, adjectives are optional
                    # remove articles in cascade
                    words = [tag[0] for tag in tree if tag[1] != "DT"]
                    postag = [tag[1] for tag in tree if tag[1] != "DT"]
                    done = False
                    i = 0
                    while True:
                        ans = paste_collapse(words[i:])
                        entities.append(ans)
                        if (postag[i] == "NN") or (i + 1 >= len(words)):
                            break
                        i += 1
        self.entities = list(set(entities))

if __name__ == "__main__":
    feedback = """
    = Livingroom =-
    You arrive in a livingroom. An usual one.

    As if things weren't amazing enough already, you can even see a sofa. The sofa is comfy. However, the sofa, like an empty sofa, has nothing on it.

    You need an exit without a door? You should try going west.

    There is a closed wooden door leading south.

    Pick the yellow bell pepper. You can eat the yellow bell pepper from the floor.

    Examine the cookbook in the counter.

    You see a closed fridge.

    Recipe #1
    --------------
    Gather all following ingredients and follow the directions to prepare this tasty meal.

    Ingredients:
        black pepper
        chicken leg
        milk
        yellow bell pepper

    Directions:
        roast the chicken leg
        prepare meal
    """
    entry = FeedbackMeta(feedback)

    feedback2 = """
    You are hungry! Go find the cookbook in the kitchen and prepare the recipe.

    = Kitchen =-
    You find yourself in a kitchen. A typical one.

    You make out a closed fridge in the corner. You rest your hand against a wall, but you miss the wall and fall onto an oven. You can make out a table. The table is massive. However, the table, like an empty table, has nothing on it. What's that over there? It looks like it's a counter. The counter is vast. On the counter you see a diced raw purple potato, a raw red potato and a cookbook. Huh, weird. As if things weren't amazing enough already, you can even see a stove. The stove is conventional. However, the stove, like an empty stove, has nothing on it. It would have been so cool if there was stuff on the stove.
    """
    entry2 = FeedbackMeta(feedback2)

    print(0)
