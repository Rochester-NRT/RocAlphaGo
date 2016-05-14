<<<<<<< HEAD
﻿import numpy as np
WHITE = -1
BLACK = +1
EMPTY = 0
PASS_MOVE = None
=======
import numpy as np
>>>>>>> latest


class TreeNode(object):
	"""Tree Representation of MCTS that covers Selection, Expansion, Evaluation, and backUp (aka 'update()')
	"""
<<<<<<< HEAD
	def __init__(self, parent, prior_p, c_puct = 5, lmbda = 0.5):
		self.lmbda = lmbda
		self.parent = parent
		self.nVisits = 1
		self.P = prior_p
		self.Q_value =  1
		self.u_value =  1#prior_p * c_puct * 10
		self.children = {}
=======
	def __init__(self, parent, prior_p):
		self.parent = parent
		self.nVisits = 0
		self.Q_value = 0
		self.u_value = prior_p
		self.children = {}
		self.P = prior_p
>>>>>>> latest

	def expansion(self, actions):
		"""Expand subtree - a dictionary with a tuple of (x,y) position as keys, TreeNode object as values

		Keyword arguments:
		Output from policy function - a list of tuples of (x, y) position and prior probability

		Returns:
		None
		"""
		for action, prob in actions:
			if action not in self.children:
<<<<<<< HEAD
				self.children[action] = TreeNode(self, prob, lmbda=self.lmbda)
=======
				self.children[action] = TreeNode(self, prob)
>>>>>>> latest

	def selection(self):
		"""Select among subtree to get the position that gives maximum action value Q plus bonus u(P)

		Keyword arguments:
		None.

		Returns:
		a tuple of (action, next_node)
		"""
<<<<<<< HEAD
		for (a, n) in self.children.iteritems():
			print (a, n.Q_value, n.u_value, n.nVisits)
=======
>>>>>>> latest
		return max(self.children.iteritems(), key=lambda (a, n): n.toValue())

	def isLeaf(self):
		"""Check if leaf node (i.e. no nodes below this have been expanded)
		"""
		return self.children == {}

	def update(self, leaf_value, c_puct):
		"""Update node values from leaf evaluation

		Arguments:
		value of traversed subtree evaluation

		Returns:
		None
		"""
		# count visit
		self.nVisits += 1
		# update Q
		mean_V = self.Q_value * (self.nVisits - 1)
		self.Q_value = (mean_V + leaf_value) / self.nVisits
		# update u (note that u is not normalized to be a distribution)
		self.u_value = c_puct * self.P * np.sqrt(self.parent.nVisits) / (1 + self.nVisits)
<<<<<<< HEAD
		print ('u_value', self.parent.nVisits, self.nVisits, self.u_value)
=======

>>>>>>> latest
	def toValue(self):
		"""Return action value Q plus bonus u(P)
		"""
		return self.Q_value + self.u_value


class MCTS(object):
	"""Monte Carlo tree search, takes an input of game state, value network function, policy network function,
	rollout policy function. get_move outputs an action after lookahead search is complete.

	The value function should take in a state and output a number in [-1, 1]
	The policy and rollout functions should take in a state and output a list of (action,prob) tuples where
		action is an (x,y) tuple

	lmbda and c_puct are hyperparameters. 0 <= lmbda <= 1 controls the relative weight of the value network and fast
	rollouts in determining the value of a leaf node. 0 < c_puct < inf controls how quickly exploration converges
	to the maximum-value policy
	"""

<<<<<<< HEAD
	def __init__(self, state, value_network, policy_network, rollout_policy, lmbda=0.5, c_puct=5, rollout_limit=500, playout_depth=20, n_search=10000, aiColor=BLACK):
		self.root = TreeNode(None, 1.0, lmbda=lmbda)
=======
	def __init__(self, state, value_network, policy_network, rollout_policy, lmbda=0.5, c_puct=5, rollout_limit=500, playout_depth=20, n_search=10000):
		self.root = TreeNode(None, 1.0)
>>>>>>> latest
		self._value = value_network
		self._policy = policy_network
		self._rollout = rollout_policy
		self._lmbda = lmbda
		self._c_puct = c_puct
		self._rollout_limit = rollout_limit
		self._L = playout_depth
		self._n_search = n_search
<<<<<<< HEAD
		self.aiColor = aiColor
=======
>>>>>>> latest

	def _DFS(self, nDepth, treenode, state):
		"""Monte Carlo tree search over a certain depth per simulation, at the end of simulation,
		the action values and visits of counts of traversed treenode are updated.

		Keyword arguments:
		Initial GameState object
		Initial TreeNode object
		Search Depth

		Returns:
		None
		"""

		visited = [None] * nDepth

		# Playout to nDepth moves using the full policy network
		for index in xrange(nDepth):
			action_probs = self._policy(state)
			# check for end of game
			if len(action_probs) == 0:
				break
			treenode.expansion(action_probs)
			action, treenode = treenode.selection()
<<<<<<< HEAD
			print action
=======
>>>>>>> latest
			state.do_move(action)
			visited[index] = treenode

		# leaf evaluation
		v = self._value(state)
<<<<<<< HEAD
		z = self._evaluate_rollout(state.copy(), self._rollout_limit)
		# when ai is WHITE, z<0 means WHITE wins
		#	Color Score EvalValue
		#   +1	  +1		+1
		#	+1	  -1		-1
		#	-1     -1       +1
		#   -1     +1       -1
		if self.aiColor == WHITE:
			z = -z
=======
		z = self._evaluate_rollout(state, self._rollout_limit)
>>>>>>> latest
		leaf_value = (1 - self._lmbda) * v + self._lmbda * z

		# update value and visit count of nodes in this traversal
		# Note: it is important that this happens from the root downward
		# so that 'parent' visit counts are correct
		for node in visited:
			node.update(leaf_value, self._c_puct)

	def _evaluate_rollout(self, state, limit):
		"""Use the rollout policy to play until the end of the game, get the winner (or 0 if tie)
		"""
		for i in xrange(limit):
			action_probs = self._rollout(state)
			if len(action_probs) == 0:
				break
			max_action = max(action_probs, key=lambda (a, p): p)[0]
			state.do_move(max_action)
		else:
			# if no break from the loop
			print "WARNING: rollout reached move limit"
<<<<<<< HEAD
		#return state.get_winner()
		return state.get_winner_Score() / 100
=======
		return state.get_winner()
>>>>>>> latest

	def get_move(self, state):
		"""After running simulations for a certain number of times, when the search is complete, an action is selected
		from root state

		Keyword arguments:
		Number of Simulations

		Returns:
		action -- a tuple of (x, y)
		"""
		action_probs = self._policy(state)
		self.root.expansion(action_probs)

		for n in xrange(0, self._n_search):
			state_copy = state.copy()
			self._DFS(self._L, self.root, state_copy)

		# chosen action is the *most visited child*, not the highest-value
		# (note that they are the same as self._n_search gets large)
		return max(self.root.children.iteritems(), key=lambda (a, n): n.nVisits)[0]

<<<<<<< HEAD
		#instead , just chosen action with the highest-value
		#return max(self.root.children.iteritems(), key=lambda (a, n): n.toValue())[0]

=======
>>>>>>> latest
	def update_with_move(self, last_move):
		"""step forward in the tree and discard everything that isn't still reachable
		"""
		if last_move in self.root.children:
			self.root = self.root.children[last_move]
			self.root.parent = None
			# siblings of root will be garbage-collected because they are no longer reachable
		else:
<<<<<<< HEAD
			self.root = TreeNode(None, 1.0, lmbda=self._lmbda)
=======
			self.root = TreeNode(None, 1.0)
>>>>>>> latest


class ParallelMCTS(MCTS):
	pass
