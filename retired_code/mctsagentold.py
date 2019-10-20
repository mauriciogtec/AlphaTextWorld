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
        self.obs = None
        self.tensor = None

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

    def cmd_history(self, cmds_only=False):
        current = self
        ans = []
        while current.parent is not None:
            index = current.index
            cmd = current.parent.edges[index].cmd
            if cmds_only:
                ans.append(cmd)
            else:
                ans.append((index, cmd))
            current = current.parent
        ans.reverse()
        return ans

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
    def __init__(self,
                 gamefile: str,
                 network: tf.keras.Model,
                 cpuct: Optional[float] = 0.4,
                 max_steps: int = 100,
                 temperature: float = 0.5):
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
        self.root.obs = self.mission
        _, _, tensor = self.network(self.mission, ["."],
                                    return_obs_tensor=True)
        self.root.tensor = tensor
        self.max_score = infos['max_score']

    def backup_edge_value(self, value: float):
        """Update to the root"""
        current = self.current
        parent = current.parent
        sum_from_leaf = value

        while parent is not None:
            edge = parent.edges[current.index]
            edge.visits += 1
            sum_from_leaf += current.score - parent.score
            edge.value += (sum_from_leaf - edge.value) / edge.visits
            current, parent = parent, parent.parent

    def backup_node_reward(self, final_ret: float):
        """Update to the root"""
        current = self.current
        parent = current.parent
        sum_from_leaf = final_ret

        while parent is not None:
            parent.visits += 1
            sum_from_leaf += current.score - parent.score

            # average rewards
            parent.reward += (sum_from_leaf - parent.reward) / parent.visits
            current, parent = parent, parent.parent

    def backup_final_ret(self, infos: dict, steps: int):
        # gamelen = steps / self.max_steps
        winfactor = infos['has_won']  # * (1.0 - 0.5 * gamelen)
        lossfactor = infos['has_lost']  # * (1.0 - 0.5 * gamelen)
        final_ret = winfactor - lossfactor
        self.backup_node_reward(final_ret)  # for learning
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
        c0 = 5
        N = sum(e.visits for e in node.edges) + c0
        eps = self.cpuct * len(node.edges) * np.sqrt(N)
        ucb = [e.value + eps * e.prior / (c0 + e.visits) for e in node.edges]

        if from_search:
            # ucb is not used, only counts
            tau = 0.01 + self.temperature *\
                (1.0 - self.current.level() / self.max_steps)

            probs = [(e.search_outcome + 0.01)**(1/tau) for e in node.edges]
            probs = np.array(probs) / sum(probs)

            # chooce proportionally
            index = np.random.choice(range(len(probs)), p=probs)
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
            msg = "EDGE {}: V: {:.2f}, P: {:.2f}, S: {}, UCB: {:.2f}, cmd: {}"
            for i, c, p, v, n, u in zip(ix, cmds, priors, values, counts, ucb):
                print(msg.format(i, v, p, n, u, c))

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
        self.current.obs = obs
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

    def available_cmds(self, infos: dict):
        node = self.current
        admissible = infos['admissible_commands']

        if 'close fridge' in admissible:
            node.extra_info['has_opened_fridge'] = True

        cmdlist = [cmd for cmd in admissible if
                   cmd != 'inventory' and
                   cmd != 'look' and
                   'insert' not in cmd and
                   'put' not in cmd and
                   'close' not in cmd and
                   'examine' not in cmd]

        if not node.extra_info['has_inventory']:
            cmdlist = [cmd for cmd in cmdlist if 'drop ' not in cmd]

        if not node.extra_info['has_examined_cookbook']:
            cmdlist = [cmd for cmd in cmdlist if
                       'take ' in cmd or
                       'open ' in cmd]
            cmdlist.extend(['go north', 'go west', 'go east', 'go south'])
            if 'examine cookbook' in admissible:
                cmdlist.append('examine cookbook')

        elif not node.extra_info['has_opened_fridge']:
            pass
            # if 'open fridge' in admissible:
            #     cmdlist = ['open fridge']

        if len(cmdlist) == 0:
            cmdlist = ['examine cookbook', 'examine fridge',
                       'go north', 'go west', 'go east', 'go south']
            node.extra_info['has_inventory'] = True  # can be game bug

        return cmdlist

    def update_node_extra_info(self):
        obs = self.current.obs
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
        obs = node.obs
        cmd = node.parent.edges[node.index].cmd
        if cmd == 'examine cookbook':
            node.score += 0.2
        elif 'examine ' in cmd:
            node.score += 0.1
        elif 'drop ' in cmd:
            if 'carrying too many things' in obs:
                node.score += 0.1
            else:
                node.score -= 0.1
        elif 'take ' in cmd and 'carrying too many things' in obs:
                node.score -= 0.1
        elif 'close ' in cmd:
            node.score -= 0.1
        elif 'put ' in cmd:
            node.score -= 0.1
        elif 'insert ' in cmd:
            node.score -= 0.1
        if cmd == 'open fridge':
            node.score += 0.2
        elif 'open ' in cmd:
            node.score += 0.1
        elif 'go ' in cmd and "can't go that way" in obs:
            node.score -= 0.05
        elif 'cook ' in cmd and "burned the" in obs:
            node.score -= 0.05
        elif 'open ' in cmd and "You have to open the" in obs:
            node.score -= 0.05
        if node.parent is not None and obs == node.parent.obs:  # repetition
            node.score -= 0.1

    def play_episode(self,
                     subtrees: int = 1,
                     max_subtree_depth: int = 8,
                     verbose: bool = False) -> Tuple:

        """play a game"""
        env, obs, infos = self.reset()
        if verbose:
            print("MISSION: ", self.mission)
            print("0. COMPUTER: ", obs)

        envscore = 0
        done = False
        num_steps = 0

        while not done:
            # span subtrees from current node as root
            # search until find a leaf or ending the game
            # subtree_memory = self.network.memory

            subtree_root = self.current

            if self.current.isleaf():
                cmdlist = self.available_cmds(infos)
                self.expand(cmdlist)

            for st in range(subtrees):
                subtree_depth = 0
                num_subtree_steps = num_steps
                while not done and subtree_depth < max_subtree_depth:
                    if self.current.isleaf():
                        cmdlist = self.available_cmds(infos)
                        self.expand(cmdlist)

                    index, cmd = self.select_move()
                    obs, envscore, done, infos = self.step(index, cmd)

                    subtree_depth += 1
                    num_subtree_steps += 1

                # subtree losses if there aren't additional points
                if not infos['has_won']:
                    infos['has_lost'] = True

                self.backup_final_ret(infos, num_subtree_steps)

                # restore root
                obs, envscore, done, infos =\
                    self.restore_checkpoint(subtree_root)

            # now select from current
            index, cmd = self.select_move(from_search=True, verbose=verbose)
            obs, envscore, done, infos = self.step(index, cmd)

            num_steps += 1

            if verbose:
                msg = "\n{}. AGENT: {}\n{}. COMPUTER: {}"
                print(msg.format(num_steps, cmd, num_steps, obs))

        if not infos['has_won']:
            infos['has_lost'] = True

        final_ret = self.backup_final_ret(infos, num_steps)
        reward = self.current.score + final_ret

        return envscore, num_steps, infos, reward

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
                cmds_node = [e.cmd for e in node.edges]
                counts_node = [e.search_outcome for e in node.edges]

                # add record to data
                memory, obs = node.obs_history(), node.obs
                record = {"cmdlist": cmds_node,
                          "counts": counts_node,
                          "reward": node.reward,
                          "obs": obs,
                          "memory": memory,
                          "level": node.level()}
                data.append(record)
        return data
