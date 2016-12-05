"""Monte Carlo Tree Search, as described in Silver et al 2015.

This is a "pure" implementation of the AlphaGo MCTS algorithm in that it is not specific to the
game of Go; everything in this file is implemented generically with respect to some state, actions,
policy function, and value function.
"""
import numpy as np
from operator import itemgetter


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        # This value for u will be overwritten in the first call to update(), but is useful for
        # choosing the first action from this node.
        self._u = prior_p
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.

        Arguments:
        action_priors -- output from policy function - a list of tuples of actions and their prior
            probability according to the policy function.

        Returns:
        None
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self):
        """Select action among children that gives maximum action value, Q plus bonus u(P).

        Returns:
        A tuple of (action, next_node)
        """
        return max(self._children.iteritems(), key=lambda act_node: act_node[1].get_value())

    def update(self, leaf_value, c_puct):
        """Update node values from leaf evaluation.

        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.

        Returns:
        None
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += (leaf_value - self._Q) / self._n_visits
        # Update u, the prior weighted by an exploration hyperparameter c_puct and the number of
        # visits. Note that u is not normalized to be a distribution.
        if not self.is_root():
            self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)

    def update_recursive(self, leaf_value, c_puct):
        """Like a call to update(), but applied recursively for all ancestors.

        Note: it is important that this happens from the root downward so that 'parent' visit
        counts are correct.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value, c_puct)
        self.update(leaf_value, c_puct)

    def get_value(self):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        """
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple (and slow) single-threaded implementation of Monte Carlo Tree Search.

    Search works by exploring moves randomly according to the given policy up to a certain
    depth, which is relatively small given the search space. "Leaves" at this depth are assigned a
    value comprising a weighted combination of (1) the value function evaluated at that leaf, and
    (2) the result of finishing the game from that leaf according to the 'rollout' policy. The
    probability of revisiting a node changes over the course of the many playouts according to its
    estimated value. Ultimately the most visited node is returned as the next action, not the most
    valued node.

    The term "playout" refers to a single search from the root, whereas "rollout" refers to the
    fast evaluation from leaf nodes to the end of the game.
    """

    def __init__(self, value_fn, policy_fn, rollout_policy_fn, lmbda=0.5, c_puct=5,
                 rollout_limit=500, playout_depth=20, n_playout=10000):
        """Arguments:
        value_fn -- a function that takes in a state and ouputs a score in [-1, 1], i.e. the
            expected value of the end game score from the current player's perspective.
        policy_fn -- a function that takes in a state and outputs a list of (action, probability)
            tuples for the current player.
        rollout_policy_fn -- a coarse, fast version of policy_fn used in the rollout phase.
        lmbda -- controls the relative weight of the value network and fast rollout policy result
            in determining the value of a leaf node. lmbda must be in [0, 1], where 0 means use only
            the value network and 1 means use only the result from the rollout.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more, and
            should be used only in conjunction with a large value for n_playout.
        """
        self._root = TreeNode(None, 1.0)
        self._value = value_fn
        self._policy = policy_fn
        self._rollout = rollout_policy_fn
        self._lmbda = lmbda
        self._c_puct = c_puct
        self._rollout_limit = rollout_limit
        self._L = playout_depth
        self._n_playout = n_playout

    def _playout(self, state, leaf_depth):
        """Run a single playout from the root to the given depth, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.

        Arguments:
        state -- a copy of the state.
        leaf_depth -- after this many moves, leaves are evaluated.

        Returns:
        None
        """
        node = self._root
        for i in range(leaf_depth):
            # Only expand node if it has not already been done. Existing nodes already know their
            # prior.
            if node.is_leaf():
                action_probs = self._policy(state)
                # Check for end of game.
                if len(action_probs) == 0:
                    break
                node.expand(action_probs)
            # Greedily select next move.
            action, node = node.select()
            state.do_move(action)

        # Evaluate the leaf using a weighted combination of the value network, v, and the game's
        # winner, z, according to the rollout policy. If lmbda is equal to 0 or 1, only one of
        # these contributes and the other may be skipped. Both v and z are from the perspective
        # of the current player (+1 is good, -1 is bad).
        v = self._value(state) if self._lmbda < 1 else 0
        z = self._evaluate_rollout(state, self._rollout_limit) if self._lmbda > 0 else 0
        leaf_value = (1 - self._lmbda) * v + self._lmbda * z

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value, self._c_puct)

    def _evaluate_rollout(self, state, limit):
        """Use the rollout policy to play until the end of the game, returning +1 if the current
        player wins, -1 if the opponent wins, and 0 if it is a tie.
        """
        player = state.get_current_player()
        for i in range(limit):
            action_probs = self._rollout(state)
            if len(action_probs) == 0:
                break
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        winner = state.get_winner()
        if winner == 0:
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.

        Arguments:
        state -- the current state, including both game state and the current player.

        Returns:
        the selected action
        """
        for n in range(self._n_playout):
            state_copy = state.copy()
            self._playout(state_copy, self._L)

        # chosen action is the *most visited child*, not the highest-value one
        # (they are the same as self._n_playout gets large).
        return max(self._root._children.iteritems(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree, assuming
        that get_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class ParallelMCTS(MCTS):
    pass
