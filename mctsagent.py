# python 3.7


import numpy as np
import tensorflow as tf
import textworld
import gym
import textworld.gym
import pickle
import attention as nn
import pdb
from textutils import *
from collections import deque

from typing import List, Union, Dict, Tuple, Optional


class Edge:
    """
    Edges contain for each action information about it's value
    """
    def __init__(self, cmd: str, prior: float, node: 'Node'):
        self.cmd = cmd
        self.prior = prior
        self.node = node
        self.value = np.random.normal(scale=1e-3)  # helps at expansion
        self.visits = 0
        self.search_outcome = 0

    def __repr__(self):
        return "Edge(N={}, S= {}, V={:.2f}, P={:.3f}, {}, {})".\
            format(self.visits, self.search_outcome,
                   self.value, self.prior,
                   self.node, self.cmd)


class Node:
    """
    Nodes save a parent state and command, and edges for valid children
    """
    def __init__(self, parent: 'Node', index: int):
        self.parent = parent
        self.index = index
        self.edges = []
        self.score = 0
        self.envscore = 0
        self.visits = 0
        self.reward = 0
        self.feedback = None
        self.inputs = None
        self.nwoutput = None

        # useful for training
        self._mainbranch = False  # set to true when played

        # training improvement
        self.extra_info = {
            'has_inventory': False,
            'has_opened_fridge': False,
            'has_examined_cookbook': False}

    def isleaf(self):
        return len(self.edges) == 0

    def addchild(self, cmd: str, prior: float):
        index = len(self.edges)
        child = Node(self, index)
        child.extra_info = self.extra_info.copy()
        edge = Edge(cmd, prior, child)
        self.edges.append(edge)

    def children(self):
        return [e.node for e in self.edges]

    def level(self):
        ans = 0
        current = self
        while current.parent is not None:
            ans += 1
            current = current.parent
        return ans

    def feedback_history(self):
        current = self
        ans = deque()
        while current is not None:
            if current.feedback.is_valid:
                ans.appendleft(current.feedback)
            current = current.parent
        return list(ans)

    def cmd_history(self, cmds_only=False):
        current = self
        ans = deque()
        while current.parent is not None:
            index = current.index
            cmd = current.parent.edges[index].cmd
            if cmds_only:
                ans.appendleft(cmd)
            else:
                ans.appendleft((index, cmd))
            current = current.parent
        return list(ans)

    def __repr__(self):
        if self.isleaf():
            return "Leaf()"
        else:
            msg = "Node(N={}, R={:.2f})"
            return msg.format(self.visits, self.reward)


class MCTSAgent:
    """
    Agents save all visited nodes, root,  MCTS related prameters,
    including the gaming neural network
    """
    VERBS = ["take", "cook", "go", "open", "drop", "slice",
             "eat", "prepare", "examine", "chop", "dice"]
    ADVERBS = ["with", "from"]
    UNWANTED_WORDS = ['a', 'an', 'the']

    def __init__(self,
                 gamefile: str,
                 network: tf.keras.Model,
                 cpuct: Optional[float] = 0.4,
                 max_steps: int = 100,
                 temperature: float = 1.0,
                 dnoise: float = 0.5,
                 verbs: List[str] = None):
        # the environment can only have ONE game
        self.gamefile = gamefile
        self.current = Node(None, None)
        self.network = network
        self.root = self.current
        self.cpuct = cpuct
        self.max_score = None
        self.max_steps = max_steps
        self.temperature = temperature
        self.vocab = network.vocab
        self.dnoise = dnoise

        infos_to_request = textworld.EnvInfos(
            description=False,
            inventory=False,
            has_won=True,
            has_lost=True,
            admissible_commands=True,
            entities=True,
            max_score=True)

        env_id = textworld.gym.register_games(
            game_files=[gamefile],
            request_infos=infos_to_request,
            max_episode_steps=max_steps)

        env = gym.make(env_id)
        self.env = env
        obs, infos = env.reset()
        self.mission = obs[1210:obs.find("=")]
        self.root.feedback = FeedbackMeta(obs[1210:])
        self.max_score = infos['max_score']

    def backup_edges(self, value: float, backup_until=None):
        """Update to the root"""
        current = self.current
        parent = current.parent
        sum_from_leaf = value

        while parent is not backup_until:
            edge = parent.edges[current.index]
            edge.visits += 1
            sum_from_leaf += current.score - parent.score
            edge.value += (sum_from_leaf - edge.value) / edge.visits
            current, parent = parent, parent.parent

    def backup_nodes(self, value: float, backup_until=None):
        """Update to the root"""
        current = self.current
        parent = current.parent
        sum_from_leaf = value

        while parent is not backup_until:
            parent.visits += 1
            sum_from_leaf += current.score - parent.score

            # average rewards
            parent.reward += (sum_from_leaf - parent.reward) / parent.visits
            current, parent = parent, parent.parent

    def backup_final_ret(self, infos: dict, steps: int, backup_until=None):
        # gamelen = steps / self.max_steps
        winfactor = 0  # infos['has_won']  # * (1.0 - 0.5 * gamelen)
        lossfactor = 0.5 * infos['has_lost']  # * (1.0 - 0.5 * gamelen)
        final_ret = winfactor - lossfactor
        self.backup_nodes(final_ret, backup_until=backup_until)
        self.backup_edges(final_ret, backup_until=backup_until)
        return final_ret

    def expand(self, inputs: dict, backup_until=None):
        """Create child for every cmd and evaluate position"""
        self.current.inputs = inputs
        cmdlist = inputs['cmdlist']
        output = self.network(inputs, training=False)

        value = output['value'].numpy().item()  # as number
        policy = tf.math.softmax(output['policy_logits']).numpy()
        self.backup_edges(value, backup_until=backup_until)

        for cmd, prior in zip(cmdlist, policy):
            self.current.addchild(cmd, prior)

    def select_move(self, from_search=False, verbose=False) -> Tuple[int, str]:
        """Select using PUCT or node count, expand for new nodes"""
        node = self.current
        c0 = 1
        N = sum(e.visits for e in node.edges) + c0
        eps = self.cpuct * len(node.edges) * np.sqrt(N)
        ucb = [e.value + eps * e.prior / (c0 + e.visits) for e in node.edges]

        if from_search:
            #
            tau = 0.01 + self.temperature *\
                (1.0 - self.current.level() / self.max_steps)

            probs = [(e.search_outcome + 0.01)**(1/tau) for e in node.edges]
            probs = np.array(probs) / sum(probs)
            dnoise = np.random.dirichlet(np.ones(len(probs)))
            eps = self.dnoise
            probs = (1.0 - eps) * probs + eps * dnoise

            # chooce proportionally
            index = np.random.choice(np.arange(len(probs)), p=probs)
            # index = np.argmax(probs)
            msg = "Chooosing {} with probability {:.4f}"
            print(msg.format(node.edges[index].cmd, probs[index]))
        else:
            node = self.current
            index = np.argmax(ucb)

        if verbose:
            msg = "NODE: Visits: {}, Reward: {:.2f}, " +\
                  "Score: {:.2f}, Envscore: {}"
            print(msg.format(
                node.visits, node.reward, node.score, node.envscore))
            # print("CMD HISTORY:", node.cmd_history(cmds_only=True))

            values = [e.value for e in node.edges]
            counts = [e.search_outcome for e in node.edges]
            cmds = [e.cmd for e in node.edges]
            priors = [e.prior for e in node.edges]
            ix = range(len(node.edges))

            msg = ''.join(("EDGE {}: V: {:.2f}, P: {:.2f}, S: {}, ",
                           "UCB: {:.2f}, cmd: {} SP: {:.2f}"))
            if not from_search:
                probs = np.zeros(len(node.edges))

            for i, c, p, v, n, u, r in zip(ix, cmds, priors,
                                           values, counts, ucb, probs):
                print(msg.format(i, v, p, n, u, c, r))

        edge = node.edges[index]
        edge.search_outcome += 1

        return index, edge.cmd

    def reset(self):
        """This one should be called instead of env.reset"""
        self.current = self.root
        env = self.env
        obs, infos = env.reset()
        obs = obs[obs.find("="):]  # removes textworld legend

        return env, obs, infos

    def close(self):
        env = self.env
        env.close()

    def step(self, index: int, cmd: str):
        """This one should be called instead of env.reset"""
        node = self.current
        node_score = node.score
        node_envscore = node.envscore
        env = self.env

        obs, envscore, done, infos = env.step(cmd)
        self.current = node.edges[index].node
        self.current.envscore = envscore
        self.current.feedback = FeedbackMeta(obs)
        self.update_node_extra_info()

        ret = envscore - node_envscore
        score = node_score + ret / self.max_score
        self.current.score = score
        self.apply_score_incentives()

        return obs, envscore, done, infos

    def restore_checkpoint(self, subtree_root: 'Node') -> str:
        """since game doesn't support copies"""
        envscore, done = 0, False
        env, obs, infos = self.reset()
        cmd_history = subtree_root.cmd_history()
        if len(cmd_history) > 0:
            for index, cmd in cmd_history:
                obs, envscore, done, infos = self.step(index, cmd)
        self.current.extra_info = subtree_root.extra_info.copy()
        return obs, envscore, done, infos

    def tokenize_from_cmd_template(self, cmd):
        words = [x for x in cmd.split() if x not in self.UNWANTED_WORDS]
        template = [words[0]]
        i = 1
        s = words[1]
        while i < len(words) - 1:
            if words[i + 1] not in self.ADVERBS:
                s += ' ' + words[i + 1]
                i += 1
            else:
                template.append(s)
                template.append(words[i + 1])
                s = words[i + 2]
                i += 2
        template.append(s)
        return template

    def get_entities(self):
        memory = self.current.feedback_history()
        entities = set(
            ["<PAD>", "<UNK>", "<S>", "</S>"] + self.VERBS + self.ADVERBS)
        for entry in memory:
            entities.update(entry.entities)
        return entities

    def get_location_and_directions(self):
        memory = self.current.feedback_history()
        locs = [x for x in memory if x.is_valid and x.is_location]
        loc = locs[-1] if len(locs) > 0 else "unknown"
        return loc.location, loc.directions, loc.entities

    def available_cmds(self, infos: dict, 
                       return_parsed_info: bool=False, verbose=False):
        node = self.current
        admissible = infos['admissible_commands']
        location, directions, loc_ents = self.get_location_and_directions()
        entities = self.get_entities()
        loc_entities = set(loc_ents)

        admissible = [cmd for cmd in admissible if  # only valid verbs
                      cmd.split()[0] in self.VERBS]

        if (  # necessary because of bug
                'examine cookbook' not in admissible and
                'cookbook' in loc_entities and
                any(['cookbook' in cmd for cmd in admissible])):
            admissible.append('examine cookbook')
        
        admissible = [cmd for cmd in admissible
                      if cmd.split()[0] != 'examine' or
                      cmd == 'examine cookbook']  # only examine this

        #                   if cmd != "examine cookbook"]

        # this shouldn't be used anymore
        # if 'close fridge' in admissible:
        #     node.extra_info['has_opened_fridge'] = True

        # we'll add directions later
        cmdlist = [cmd for cmd in admissible if 'go ' not in cmd]

        # remove commands that have unseen entities
        # this is a bug in textworld, to minimize the impact we do two things
        # remove with respect to entities and not local entities which should
        parent = node.parent
        if parent is None or not parent.extra_info['has_examined_cookbook']:
            cmdlist = [cmd for cmd in cmdlist if
                       cmd.split()[0] in ("examine", "go", "open", "take")]

        tmp = []
        for cmd in cmdlist:
            words = self.tokenize_from_cmd_template(cmd)
            if words[1] in entities and (len(words) < 4 or words[3] in entities):
                tmp.append(cmd)
        cmdlist = tmp

        # add valid directions
        for d in directions:
            cmdlist.append("go " + d)

        # if not node.extra_info['has_inventory']:
        #     cmdlist = [cmd for cmd in cmdlist if 'drop ' not in cmd]

        # if not node.extra_info['has_examined_cookbook']:
        #     cmdlist = [cmd for cmd in cmdlist
        #                if cmd == 'examine cookbook' or
        #                'examine' != cmd.split()[0]]

        # elif not node.extra_info['has_opened_fridge']:
        #     pass
            # if 'open fridge' in admissible:
            #     cmdlist = ['open fridge']

        if not return_parsed_info:
            return cmdlist
        else:
            return cmdlist, entities, location, directions

    def update_node_extra_info(self):
        obs = self.current.feedback.text
        cmd = self.current.parent.edges[self.current.index].cmd
        if "carrying too many things" in obs or 'take ' in cmd:
            self.current.extra_info['has_inventory'] = True
        if cmd == 'open fridge':
            self.current.extra_info['has_opened_fridge'] = True
        if cmd == 'examine cookbook':
            self.current.extra_info['has_examined_cookbook'] = True

    def apply_score_incentives(self):
        # penalize droping and examining
        node = self.current
        examined_cookbook = (
            False if node.parent is None else
            node.parent.extra_info['has_examined_cookbook'])
        obs = node.feedback.text
        cmd = node.parent.edges[node.index].cmd
        verb = cmd.split()[0]
        if cmd == 'examine cookbook' and not examined_cookbook:
            node.score += 0.2
        elif cmd != 'examine cookbook' and verb == 'examine':
            node.score -= 0.1
        elif verb == 'drop':
            if 'carrying too many things' in obs:
                node.score += 0.1
            else:
                node.score -= 0.1
        if cmd == 'open fridge':
            node.score += 0.1
        elif verb == 'open':
            node.score += 0.1
        elif verb == 'cook' and "burned the" in obs:
            node.score -= 0.1

    def get_tensor_inputs(self, infos: dict, verbose=False):
        inputs = dict()
        cmdlist, entities, location, directions = self.available_cmds(
            infos, return_parsed_info=True, verbose=verbose)
        inputs['cmdlist'] = cmdlist
        entities = list(entities)
        word2id = {w: i for i, w in enumerate(self.vocab)}
        ents2id = {w: i for i, w in enumerate(entities)}
        # memory
        memory_texts = []
        memory = self.current.feedback_history()
        for x in memory:
            memory_texts.extend(x.sentences)
        meminputs = text2tensor(memory_texts, word2id)
        inputs['memory_input'] = tf.constant(meminputs, tf.int32)
        # location
        locinputs = get_word_id(location, word2id)
        inputs['location_input'] = tf.constant([locinputs], tf.int32)
        # commands
        cmdinputs = text2tensor(cmdlist, word2id)
        inputs['cmdlist_input'] = tf.constant(cmdinputs, tf.int32)
        # ent vocab
        entvocab = text2tensor(entities, word2id)
        inputs['entvocab_input'] = tf.constant(entvocab, tf.int32)
        # next word inp uts
        nwinputs = []
        nwoutput = []
        for cmd in cmdlist:
            tokens = self.tokenize_from_cmd_template(cmd)
            tokens = ["<S>"] + tokens + ["</S>"]
            tokens = [get_word_id(w, ents2id) for w in tokens]
            for i in range(1, len(tokens)):
                nwinputs.append(tokens[:i])
                nwoutput.append(tokens[i])
        maxlen = max([len(t) for t in nwinputs])
        pad = ents2id["<PAD>"]
        nwinputs = [t + [pad] * (maxlen - len(t)) for t in nwinputs]
        inputs['cmdprev_input'] = tf.constant(np.array(nwinputs), tf.int32)
        inputs['ents2id'] = ents2id

        self.current.nwoutput = nwoutput

        return inputs

    def play_episode(self,
                     subtrees: int = 1,
                     max_subtree_depth: int = 8,
                     verbose: bool = False) -> Tuple:

        """play a game"""
        env, obs, infos = self.reset()
        self.current._mainbranch = True

        if verbose:
            print("MISSION: ", self.mission)
            print("0. COMPUTER: ", obs)

        envscore = 0
        done = False
        num_steps = 0

        while not done:
            # span subtrees from current node as root
            # search until find a new point or ending the game
            subtree_root = self.current
            if self.current.isleaf():
                inputs = self.get_tensor_inputs(infos, verbose=verbose)
                self.expand(inputs)

            for st in range(subtrees):
                subtree_depth = 0
                num_subtree_steps = num_steps
                while not done and subtree_depth < max_subtree_depth:
                    if self.current.isleaf():
                        inputs = self.get_tensor_inputs(infos, verbose=verbose)
                        self.expand(inputs)

                    index, cmd = self.select_move()
                    obs, envscore, done, infos = self.step(index, cmd)

                    subtree_depth += 1
                    num_subtree_steps += 1

                self.backup_final_ret(infos, num_subtree_steps)

                # restore root
                obs, envscore, done, infos =\
                    self.restore_checkpoint(subtree_root)

            # now select from current
            index, cmd = self.select_move(from_search=True, verbose=verbose)
            obs, envscore, done, infos = self.step(index, cmd)
            self.current._mainbranch = True

            num_steps += 1

            if verbose:
                msg = "\n{}. AGENT: {}\n{}. COMPUTER: {}"
                print(msg.format(num_steps, cmd, num_steps, obs))

        final_ret = self.backup_final_ret(infos, num_steps)
        reward = self.current.score + final_ret

        return envscore, num_steps, infos, reward

    def dump_tree(self, mainbranch: bool=True) -> List:
        """simple tree traversal for dumping data"""
        data = []

        tovisit = [self.root]
        while len(tovisit) > 0:
            node = tovisit.pop(0)
            if not node.isleaf():
                if not mainbranch or node._mainbranch:
                    # add node info to record
                    tovisit.extend(node.children())

                    # extract edge data
                    cmds_node = [e.cmd for e in node.edges]
                    counts_node = [e.search_outcome for e in node.edges]

                    # add record to data
                    feedback_history = node.feedback_history()
                    inputs = dict()
                    inputs['cmdlist'] = node.inputs['cmdlist']
                    inputs['ents2id'] = {k: i for k, i in node.inputs['ents2id'].items()}
                    inputs['memory_input'] = node.inputs['memory_input'].numpy().tolist()
                    inputs['location_input'] = node.inputs['location_input'].numpy().tolist()
                    inputs['cmdlist_input'] = node.inputs['cmdlist_input'].numpy().tolist()
                    inputs['entvocab_input'] = node.inputs['entvocab_input'].numpy().tolist()
                    inputs['cmdprev_input'] = node.inputs['cmdprev_input'].numpy().tolist()
                    
                    record = {
                        "cmdlist": cmds_node,
                        "inputs": inputs,
                        "nwoutput": node.nwoutput,
                        "counts": counts_node,
                        "value": node.reward,
                        "feedback_history": feedback_history,
                        "feedback_meta": node.feedback,
                        "level": node.level(),
                        "mainbranch": node._mainbranch}
                data.append(record)
        return data
