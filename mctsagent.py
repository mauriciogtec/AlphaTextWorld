# Assumes python 3.7
# Assumes a pretrained neural network model saved as keras model

import numpy as np
import tensorflow as tf
import textworld
import gym
import textworld.gym
import pickle
import neuralnetwork
import pdb

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

    def __repr__(self):
        return "Edge(visits={}, value={:.2f}, prior={:.3f}, node={})".\
            format(self.visits, self.value, self.prior, self.node)


class Node:
    """
    Nodes save a parent state and command, and edges for valid children
    """
    def __init__(self, parent: 'Node', index: int):
        self.parent = parent
        self.index = index
        self.edges = []
        self.score = 0
        self.visits = 0
        self.reward = 0
        self.obs = None
        self.tensor = None

    def isleaf(self):
        return len(self.edges) == 0

    def addchild(self, cmd: str, prior: float):
        index = len(self.edges)
        child = Node(self, index)
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

    def obs_history(self):
        current = self
        ans = []
        while current is not None:
            ans.append(current.obs)
            current = current.parent
        ans.reverse()
        return ans

    def tensor_history(self):
        current = self
        ans = []
        while current is not None:
            if current.tensor is not None:
                ans.append(current.tensor)
            current = current.parent
        ans.reverse()
        return ans

    def cmd_history(self):
        current = self
        ans = []
        while current.parent is not None:
            index = current.index
            cmd = current.parent.edges[index].cmd
            ans.append((index, cmd))
            current = current.parent
        ans.reverse()
        return ans

    def __repr__(self):
        if self.isleaf():
            return "Leaf()"
        else:
            msg = "Node(visits={}, reward={:.2f})"
            return msg.format(self.visits, self.reward)


class MCTSAgent:
    """
    Agents save all visited nodes, root,  MCTS related prameters,
    including the gaming neural network
    """
    def __init__(self,
                 gamefile: str,
                 network: tf.keras.Model,
                 cpuct: Optional[float] = 0.4,
                 max_steps: int = 100,
                 temperature: float = 1.0):
        # the environment can only have ONE game
        self.gamefile = gamefile
        self.current = Node(None, None)
        self.network = network
        self.root = self.current
        self.cpuct = cpuct
        self.max_score = None
        self.max_steps = max_steps
        self.temperature = temperature

        infos_to_request = textworld.EnvInfos(
            description=False,
            inventory=False,
            has_won=True,
            has_lost=True,
            admissible_commands=True,
            max_score=True)
        env_id = textworld.gym.register_games(
            game_files=[gamefile],
            request_infos=infos_to_request,
            max_episode_steps=max_steps)

        env = gym.make(env_id)
        self.env = env
        obs, infos = env.reset()
        self.mission = obs[1210:obs.find("=")]
        self.root.obs = self.mission
        _, _, tensor = self.network(self.mission, ["."],
                                    return_obs_tensor=True)
        self.root.tensor = tensor
        self.max_score = infos['max_score']

    def backup_edge_value(self, value: float):
        """Update to the root"""
        current = self.current
        parent = current.parent
        accrue = 0

        while parent is not None:
            edge = parent.edges[current.index]
            edge.visits += 1
            accrue += current.score - parent.score
            edge.value += (accrue + value - edge.value) / edge.visits
            current, parent = parent, parent.parent

    def backup_node_reward(self, final_ret: float):
        """Update to the root"""
        current = self.current
        parent = current.parent
        accrue = 0

        while parent is not None:
            parent.visits += 1
            accrue += current.score - parent.score
            parent.reward += \
                (accrue + final_ret - parent.reward) / parent.visits
            current, parent = parent, parent.parent

    def backup_final_ret(self, infos: dict, steps: int):
        gamelen = steps / self.max_steps
        winfactor = infos['has_won'] * (1.0 - 0.5 * gamelen)
        lossfactor = infos['has_lost'] * (1.0 - 0.5 * gamelen)
        final_ret = winfactor - lossfactor
        self.backup_node_reward(final_ret)
        self.backup_edge_value(final_ret)  # improve current gameplay
        return final_ret

    def expand(self, cmdlist: str):
        """Create child for every cmd and evaluate position"""
        value, policy, obstensor =\
            self.network(self.current.obs, cmdlist,
                         memory=self.current.tensor_history(),
                         tensor_memory=True,
                         return_obs_tensor=True)
        self.current.tensor = obstensor

        value = value.numpy().item()  # as number
        self.backup_edge_value(value)

        for cmd, prior in zip(cmdlist, policy):
            self.current.addchild(cmd, prior)

    def select_move(self, from_search=False, verbose=False) -> Tuple[int, str]:
        """Select using PUCT or node count, expand for new nodes"""
        node = self.current
        N = sum(e.visits for e in node.edges)  # TODO: possible bug here ????
        eps = self.cpuct * len(node.edges) * np.sqrt(N)
        ucb = [e.value + eps * e.prior / (1 + e.visits) for e in node.edges]
        if from_search:  # ucb is not used, only counts
            tau = 0.01 + (1.0 / self.temperature) *\
                (1.0 - self.current.level() / self.max_steps)
            probs = np.array([(e.visits + 1)**(1 / tau) for e in node.edges])
            probs /= probs.sum()
            index = np.random.choice(range(len(probs)), p=probs)
        else:
            node = self.current
            index = np.argmax(ucb)

        if verbose:
            values = [e.value for e in node.edges]
            visits = [e.visits for e in node.edges]
            cmds = [e.cmd for e in node.edges]
            ix = range(len(node.edges))
            msg = "{}: val: {:.2f}, N: {}, UCB: {:.2f}, cmd: {}"
            for i, c, v, n, u in zip(ix, cmds, values, visits, ucb):
                print(msg.format(i, v, n, u, c))

        return index, node.edges[index].cmd

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
        env = self.env
        obs, score, done, infos = env.step(cmd)
        self.current = self.current.edges[index].node
        self.current.obs = obs
        return obs, score, done, infos

    def restore_checkpoint(self, subtree_root: 'Node') -> str:
        """since game doesn't support copies"""
        score, done = 0, False
        env, obs, infos = self.reset()
        cmd_history = subtree_root.cmd_history()
        if len(cmd_history) > 0:
            for index, cmd in cmd_history:
                try:
                    obs, score, done, infos = self.step(index, cmd)
                except:
                    pdb.set_trace()
        return obs, score, done, infos

    def avail_cmds(self, infos: dict):
        cmdlist = [cmd for cmd in infos['admissible_commands']
                   if cmd not in ['inventory', 'look']]
        return cmdlist

    def play_episode(self,
                     subtrees: int = 1,
                     max_subtree_depth: int = 8,
                     verbose: bool = False) -> Tuple:

        """play a game"""
        env, obs, infos = self.reset()
        if verbose:
            print("MISSION: ", self.mission)
            print("0. COMPUTER: ", obs)

        score = 0
        done = False
        num_steps = 1

        while not done:
            # span subtrees from current node as root
            # search until find a leaf or ending the game
            # subtree_memory = self.network.memory

            subtree_root = self.current

            if self.current.isleaf():
                cmdlist = self.avail_cmds(infos)
                self.expand(cmdlist)

            for st in range(subtrees):
                subtree_depth = 0
                num_subtree_steps = num_steps
                while not done and subtree_depth < max_subtree_depth:
                    if self.current.isleaf():
                        cmdlist = self.avail_cmds(infos)
                        self.expand(cmdlist)
                    index, cmd = self.select_move()
                    obs, score, done, infos = self.step(index, cmd)
                    self.current.score = score / self.max_score
                    subtree_depth += 1
                    num_subtree_steps += 1

                # backup final return if done, expand if leaf
                if done:
                    self.backup_final_ret(infos, num_subtree_steps)

                # restore root
                obs, score, done, infos =\
                    self.restore_checkpoint(subtree_root)

            # now select from current
            index, cmd = self.select_move(from_search=True, verbose=verbose)
            obs, score, done, infos = self.step(index, cmd)
            self.current.score = score / self.max_score
            num_steps += 1

            if verbose:
                print("{}. AGENT: {}".format(num_steps, cmd))
                print("{}. COMPUTER: {}".format(num_steps, obs))

        final_ret = self.backup_final_ret(infos, num_steps)
        reward = score + final_ret

        return score, num_steps - 1, infos, reward

    def dump_tree(self) -> Dict:
        """simple tree traversal for dumping data"""
        data = []

        tovisit = [self.root]
        while len(tovisit) > 0:
            node = tovisit.pop(0)
            if not node.isleaf():
                # add node info to record
                tovisit.extend(node.children())

                # extract edge data
                cmds_node = []
                counts_node = []
                for edge in node.edges:
                    cmds_node.append(edge.cmd)
                    counts_node.append(edge.visits)

                # add record to data
                memory, obs = node.obs_history(), node.obs
                record = {"cmdlist": cmds_node,
                          "counts": counts_node,
                          "reward": node.reward,
                          "obs": obs,
                          "memory": memory}
                data.append(record)
        return data
