
"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
# Monte_carlo.py

from collections import defaultdict
import math
import random

class MCTS:
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)         # total reward of each state key
        self.N = defaultdict(int)         # visit count for each state key
        self.children = dict()            # maps state key to a set of child state keys
        self.exploration_weight = exploration_weight
        self.state_map = {}               # maps state keys back to states

    def _state_key(self, state):
        to_move = state.to_move
        board_key = tuple(state.board.items())
        moves_key = tuple(state.moves)
        return (to_move, board_key, moves_key)

    def choose(self, game, state):
        if game.terminal_test(state):
            raise RuntimeError("Cannot choose from a terminal state.")

        s_key = self._state_key(state)
        if s_key not in self.children:
            raise RuntimeError("State has no children. Run rollouts first.")

        def score(c_key):
            if self.N[c_key] == 0:
                return float('-inf')
            return self.Q[c_key] / self.N[c_key]

        # Pick the child state key with the best score
        best_child_key = max(self.children[s_key], key=score)
        # Return the actual state corresponding to the best child key
        return self.state_map[best_child_key]

    def do_rollout(self, game, state):
        s_key = self._state_key(state)
        self.state_map[s_key] = state
        path = self._select(game, s_key)
        leaf_key = path[-1]
        self._expand(game, leaf_key)
        reward = self._simulate(game, leaf_key)
        self._backpropagate(path, reward)

    def _select(self, game, s_key):
        path = []
        while True:
            path.append(s_key)
            if s_key not in self.children or not self.children[s_key]:
                # Leaf node or unexpanded node
                return path
            # Check if there's any unexplored child
            unexplored = self.children[s_key] - self.children.keys()
            if unexplored:
                n_key = unexplored.pop()
                path.append(n_key)
                return path
            # All children expanded, choose UCT child
            s_key = self._uct_select(s_key)

    def _expand(self, game, s_key):
        state = self.state_map[s_key]
        if s_key in self.children:
            return  # already expanded
        if game.terminal_test(state):
            self.children[s_key] = set()  # no children
        else:
            child_keys = set()
            for move in game.actions(state):
                next_state = game.result(state, move)
                n_key = self._state_key(next_state)
                self.state_map[n_key] = next_state
                child_keys.add(n_key)
            self.children[s_key] = child_keys

    def _simulate(self, game, s_key):
        state = self.state_map[s_key]
        while not game.terminal_test(state):
            moves = game.actions(state)
            move = random.choice(moves)
            state = game.result(state, move)
        # Utility wrt the initial state's first player
        return game.utility(state, game.to_move(game.initial))

    def _backpropagate(self, path, reward):
        for s_key in reversed(path):
            self.N[s_key] += 1
            self.Q[s_key] += reward
            # Invert reward for the parent level
            reward = 1 - reward

    def _uct_select(self, s_key):
        # All children of s_key should be expanded
        log_N_vertex = math.log(self.N[s_key])
        def uct(c_key):
            return (self.Q[c_key]/self.N[c_key]) + \
                   self.exploration_weight * math.sqrt(log_N_vertex / self.N[c_key])
        return max(self.children[s_key], key=uct)
