


# deprecated code
 
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import activations, backend,\
#     optimizers, utils, layers, models, regularizers, initializers
# import sys
# from textutils import *
# from custom_layers import *
# # sys.path.append("../")


# def AlphaTextWorldNet(embeddings, vocab):
#     """
#     Learn to play from memory
#     """
#     REG_PENALTY = 1e-5
#     KSIZE = 3
#     HIDDEN_UNITS = 64
#     ATT_HEADS = 4
#     POSFREQS = 16
#     MAX_CMD_LEN = 6

#     # 1. Define Inputs
#     meminput = layers.Input(
#         shape=(None, ),  # M X Tm
#         name="memory")
#     cmdinput = layers.Input(
#         shape=(None, ),  # C X Tc
#         name="cmdlist")
#     locinput = layers.Input(
#         shape=(1, ),  # 1 X 1
#         batch_size=1,
#         name="location")
#     nwinput = layers.Input(
#         shape=(None, ),  # C x Tr
#         name="nextwords")
#     entvocab = layers.Input(
#         shape=(None, ),  # V x Tv
#         name="entvocab")
#     inputs = [meminput, cmdinput, locinput, nwinput, entvocab]

#     # 2. Core Layers
#     word2id = {w: i for i, w in enumerate(vocab)}
#     id2word = {i: w for i, w in word2id.items()}
#     embedding_dim, vocab_size = embeddings.shape
#     embedding = layers.Embedding(
#         input_dim=vocab_size,
#         input_length=None,
#         output_dim=embedding_dim,
#         embeddings_initializer=initializers.Constant(embeddings),
#         trainable=True)
#     lfe_memory = LocalFeaturesExtractor(
#         filters=HIDDEN_UNITS,
#         kernel_size=KSIZE,
#         num_blocks=2,
#         l2=REG_PENALTY,
#         name="lfe_memory")
#     lfe_cmdlist = LocalFeaturesExtractor(
#         filters=HIDDEN_UNITS,
#         kernel_size=KSIZE,
#         num_blocks=1,
#         l2=REG_PENALTY,
#         name="lfe_cmdlist")
#     att_memory_loc_time = AttentionEncoder(
#         units=HIDDEN_UNITS,
#         num_heads=ATT_HEADS,
#         num_blocks=1,
#         l2=REG_PENALTY,
#         name="att_memory_loc_time")
#     att_memory_loc_turn = AttentionEncoder(
#         units=HIDDEN_UNITS,
#         num_heads=ATT_HEADS,
#         num_blocks=1,
#         l2=REG_PENALTY,
#         name="att_memory_loc_turn")

#     # 3. Network Flow
#     memx = embedding(meminput)  # M X Tr X D
#     memx = lfe_memory(memx)  # M x T x D
#     cmdx = embedding(cmdinput)  # C X Tc X D
#     cmdx = lfe_cmdlist(cmdx)  # C X Tc X D
#     locx = embedding(locinput)  # 1 X 1 X D
#     locx = tf.squeeze(locx, 0)  # 1 x D
#     nwx = embedding(nwinput)  # C X Tr X D
#     evx = embedding(entvocab)  # V X Tv X D

#     # queryx = tf.math.reduce_sum(cmdx, axis=1)  # C x D

#     # 1. pipeline for value prediction
#     memlocx = att_memory_loc_time(locx, memx)  # M x 1 x D
#     memlocx = tf.transpose(memlocx, perm=(1, 0, 2))  # 1 X M x D
#     memlocx = tf.identity(memlocx, name="hello")  # 1 X M x D
#     # memlocx = att_memory_loc_turn(locx, memlocx)  # 1 x 1 x D

#     outputs = [locx, memlocx]

#     model = models.Model(
#         inputs=inputs,
#         outputs=outputs)

#     return model


# if  __name__ == "__main__":

#     VERBS = ["take", "cook", "go", "open", "drop",
#              "eat", "prepare", "examine", "chop", "dice"]
#     ADVERBS = ["with", "from"]

#     path = '/home/mauriciogtec/'
#     textworld_vocab = set()
#     with open(path + 'Github/TextWorld/montecarlo/vocab.txt', 'r') as fn:
#         for line in fn:
#             word = line[:-1]
#             textworld_vocab.add(word)

#     embedding_dim = 100
#     embedding_fdim = 64
#     embeddings, vocab = load_embeddings(
#         embeddingsdir=path + "glove.6B/",
#         embedding_dim=embedding_dim,  # try 50
#         vocab=textworld_vocab)
#     index = np.random.permutation(range(embedding_dim))[:embedding_fdim]
#     embeddings = embeddings[index, :]

#     feedback = """
#     = Livingroom =-
#     You arrive in a livingroom. An usual one.

#     As if things weren't amazing enough already, you can even see a sofa. The sofa is comfy. However, the sofa, like an empty sofa, has nothing on it.

#     You need an exit without a door? You should try going west.

#     There is a closed wooden door leading south.

#     Pick the yellow bell pepper. You can eat the yellow bell pepper from the floor.
#     """
#     feedback2 = """
#     You are hungry! Go find the cookbook in the kitchen and prepare the recipe.

#     = Kitchen =-
#     You find yourself in a kitchen. A typical one.

#     You make out a closed fridge in the corner. You rest your hand against a wall, but you miss the wall and fall onto an oven. You can make out a table. The table is massive. However, the table, like an empty table, has nothing on it. What's that over there? It looks like it's a counter. The counter is vast. On the counter you see a diced raw purple potato, a raw red potato and a cookbook. Huh, weird. As if things weren't amazing enough already, you can even see a stove. The stove is conventional. However, the stove, like an empty stove, has nothing on it. It would have been so cool if there was stuff on the stove.
#     """

#     entry1 = FeedbackMeta(feedback)
#     entry2 = FeedbackMeta(feedback2)

#     memory = [entry1, entry2]

    # PREPARE INPUTS ----------------------------
    # memory
    word2id = {w: i for i, w in enumerate(vocab)}
    memory_texts = []
    for x in memory:
        if x.is_valid:
            memory_texts.extend(x.sentences)
    meminputs = text2tensor(memory_texts, word2id)
    # location
    locs = [x for x in memory if x.is_valid and x.is_location]
    location = locs[-1].location if len(locs) > 0 else "unknown"
    locinputs = get_word_id(location, word2id)
    # commands
    cmdlist = ["go south", "open fridge", "examine cookbook"]
    cmdinputs = text2tensor(cmdlist, word2id)
    # ent vocab
    entities = ["<PAD>", "<UNK>", "<S>", "</S>"] + VERBS + ADVERBS
    entities = set(entities)
    for x in memory:
        if x.is_valid:
            entities.update(x.entities)
    entities = list(entities)
    entinputs = text2tensor(entities, word2id)
    # next word inputs
    nwin = []
    nwout = []
    entinputs2id = {w: i for i, w in enumerate(entities)}
    for cmd in cmdlist:
        tokens = tokenize_from_cmd_template(cmd)
        tokens = ["<S>"] + tokens + ["</S>"]
        tokens = [get_word_id(w, entinputs2id) for w in tokens]
        for i in range(1, len(tokens)):
            nwin.append(tokens[:i])
            nwout.append(tokens[i])
    maxlen = max([len(t) for t in nwin])
    pad = entinputs2id["<PAD>"]
    rows = [t + [pad] * (maxlen - len(t)) for t in nwin]
    nwin = np.array(rows, dtype=int)
    nwout = np.array(nwout, dtype=int)

    # GENERATE NETWORK
    model = AlphaTextWorldNet(embeddings, vocab)
    model.summary()

    print(0)
