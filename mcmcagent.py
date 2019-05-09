# Assumes python 3.7
# Assumes a pretrained neural network model saved as keras model

import numpy as np
import tensorflow as tf
import textworld
import gym
import textworld.gym
import pickle
import neuralnetwork

from typing import List, Union, Dict, Tuple, Optional

class Edge:
    """
    Edges contain for each action information about it's value
    """
    def __init__(self, cmd: str, prior: float, node: 'Node'):
        self.cmd = cmd
        self.prior = prior
        self.node = node
        self.value = np.random.normal(scale=1e-4) # helps at expansion
        self.visits = 0

    def __repr__(self):
        return "Edge(visits={}, value={:.2f}, prior={:.3f})".format(self.visits, self.value, self.prior)

class Node:
    """
    Nodes save a parent state and command, and edges for valid children
    """
    def __init__(self, parent: 'Node', index: int):
        self.parent = parent
        self.index = index
        self.edges = []
        self.obs = None
        self.last_score = 0

    def isleaf(self):
        return len(self.edges) == 0

    def addchild(self, cmd: str, prior: float):
        index = len(self.edges)
        child = Node(self, index)
        edge = Edge(cmd, prior, child)
        self.edges.append(edge)

    def children(self):
        return [e.node for e in self.edges]

    def __repr__(self):
        visits = sum(e.visits for e in self.edges)
        value = max(e.value for e in self.edges)
        return "Node(visits={}, value={:.2f})".format(visits, value)

class MCTSAgent:
    """
    Agents save all visited nodes, root,  MCTS related prameters,
    including the gaming neural network
    """

    def __init__(self, gamefile: str, network: tf.keras.Model, cpuct: Optional[float] = 0.4):
        self.gamefile = gamefile # the environment can only have ONE game registered
        self.current = Node(None, None)
        self.network = network
        self.root = self.current
        self.cpuct = cpuct

    @property
    def env(self) -> textworld.Environment:
        """Register environment from game file"""
        infos_to_request = textworld.EnvInfos(
            description=True, 
            has_won=True, 
            has_lost=True,
            inventory=False, 
            admissible_commands=True,
            max_score=True)
        env_id = textworld.gym.register_games(
            game_files=[self.gamefile],
            request_infos=infos_to_request,
            max_episode_steps=100)
        env = gym.make(env_id)
        return env
    
    def backup(self, value: float):
        """Update to the root"""
        current = self.current
        parent = current.parent
        accrue = 0

        while parent is not None:
            edge = parent.edges[current.index]
            edge.visits += 1
            accrue += current.last_score - parent.last_score
            edge.value += (accrue + value - edge.value) / edge.visits
            current = parent
            parent = current.parent

    def expand(self, obs: str, cmdlist: str):
        """Create child for every cmd and evaluate position"""
        value, policy = self.network(obs, cmdlist)
        self.network.memory.append(obs)

        self.current.obs = obs

        value = value.numpy().item() # as number
        self.backup(value)

        for cmd, prior in zip(cmdlist, policy):
            self.current.addchild(cmd, prior)

    def select_move(self, obs: str, cmdlist: str) -> int:
        """Select using PUCT, expand for new nodes"""
        node = self.current

        if node.isleaf():
            self.expand(obs, cmdlist)

        N = sum(e.visits for e in node.edges) + 1
        ucb = [e.value + self.cpuct * np.sqrt(N) / (1 + e.visits) for e in node.edges]
        index = np.argmax(ucb)

        return index
    
    def restore_root(self):
        self.current = self.root
        self.current.last_score = 0
        self.network.reset_states()

    def advance_edge(self, index: int):
        self.current = self.current.edges[index].node

    def dump_data(self) -> Dict:
        """simple bfs traversal dumping data"""
        obs = []
        values = []
        cmds = []
        probs = []
        counts = []
        
        tovisit = [self.root]
        while len(tovisit) > 0:
            node = tovisit.pop()
            if not node.isleaf():
                # add node info to record
                obs.append(node.obs)
                values.append(node.value)
                tovisit.extend(node.children())
               
                # extract node values and probs
                cmds_node = []
                probs_node = []
                counts_node = []

                for edge in node.edges:
                    probs_node.append(edge.prior)
                    cmds_node.append(edge.cmd)
                    counts.append(edge.visits)

                cmds.append(cmds_node)
                probs.append(probs_node)
                counts.append(counts_node)

        data = {'values': values, 'cmds': cmds, 'probs': probs}
        
        return data

    def play_game(self) -> Tuple:
        """play a game"""
        self.restore_root()
        env = self.env
        obs, infos = env.reset()
        self.network.memory = []

        done = False
        num_moves = 0

        while not done:
            num_moves += 1
            # admissible commands are necessary for training, todo: save them to train command prediction
            cmdlist = [cmd for cmd in infos['admissible_commands'] if cmd not in {'inventory', 'look'}]
            index = self.select_move(obs, cmdlist)
            self.advance_edge(index)

            # advance the name
            obs, score, done, infos = env.step(cmdlist[index])
            self.current.last_score = score

        reward = float(infos['has_won']) * (150 - num_moves) - float(infos['has_lost'])
        self.backup(reward) 

        return score, num_moves, infos
