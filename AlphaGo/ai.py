"""Policy players"""
import numpy as np
from AlphaGo import go
from AlphaGo import mcts


class GreedyPolicyPlayer(object):
	"""A player that uses a greedy policy (i.e. chooses the highest probability
	move each turn)
	"""

	def __init__(self, policy_function):
		self.policy = policy_function

	def get_move(self, state):
		sensible_moves = [move for move in state.get_legal_moves() if not state.is_eye(move, state.current_player)]
		if len(sensible_moves) > 0:
			move_probs = self.policy.eval_state(state, sensible_moves)
			max_prob = max(move_probs, key=lambda (a, p): p)
			return max_prob[0]
		# No 'sensible' moves available, so do pass move
		return go.PASS_MOVE


class ProbabilisticPolicyPlayer(object):
	"""A player that samples a move in proportion to the probability given by the
	policy.

	By manipulating the 'temperature', moves can be pushed towards totally random
	(high temperature) or towards greedy play (low temperature)
	"""

	def __init__(self, policy_function, temperature=1.0):
		assert(temperature > 0.0)
		self.policy = policy_function
		self.beta = 1.0 / temperature

	def get_move(self, state):
		sensible_moves = [move for move in state.get_legal_moves() if not state.is_eye(move, state.current_player)]
		if len(sensible_moves) > 0:
			move_probs = self.policy.eval_state(state, sensible_moves)
			# zip(*list) is like the 'transpose' of zip; zip(*zip([1,2,3], [4,5,6])) is [(1,2,3), (4,5,6)]
			moves, probabilities = zip(*move_probs)
			probabilities = np.array(probabilities)
			probabilities = probabilities ** self.beta
			probabilities = probabilities / probabilities.sum()
			# numpy interprets a list of tuples as 2D, so we must choose an _index_ of moves then apply it in 2 steps
			choice_idx = np.random.choice(len(moves), p=probabilities)
			return moves[choice_idx]
		return go.PASS_MOVE

	def get_moves(self, states):
		"""Batch version of get_move. A list of moves is returned (one per state)
		"""
		sensible_move_lists = [[move for move in st.get_legal_moves() if not st.is_eye(move, st.current_player)] for st in states]
		all_moves_distributions = self.policy.batch_eval_state(states, sensible_move_lists)
		move_list = [None] * len(states)
		for i, move_probs in enumerate(all_moves_distributions):
			if len(move_probs) == 0:
				move_list[i] = go.PASS_MOVE
			else:
				# this 'else' clause is identical to ProbabilisticPolicyPlayer.get_move
				moves, probabilities = zip(*move_probs)
				probabilities = np.array(probabilities)
				probabilities = probabilities ** self.beta
				probabilities = probabilities / probabilities.sum()
				choice_idx = np.random.choice(len(moves), p=probabilities)
				move_list[i] = moves[choice_idx]
		return move_list


class MCTSPlayer(object):
	def __init__(self, policy_function, value_function, rollout_function, lmbda=.5, c_puct=5, rollout_limit=500, playout_depth=40, n_search=100):
		self.policy_function = policy_function
		self.value_function = value_function
		self.rollout_function = rollout_function
		self._lmbda = lmbda
		self._c_puct = c_puct
		self._rollout_limit = rollout_limit
		self._playout_depth = playout_depth
		self._n_search = n_search

		self.mcts = None

	def get_move(self, state):
		# should call update_with_move so we can benefit from tree from earlier move, revisit because saw error
		# For now will instantiate each call
		self.mcts = mcts.MCTS(state, self.value_function, self.policy_function, self.rollout_function, self.lmbda, self.c_puct,
					self.rollout_limit, self.playout_depth, self.n_search)

		sensible_moves = [move for move in state.get_legal_moves() if not state.is_eye(move, state.current_player)]
		if len(sensible_moves) > 0:
			# self.mcts.update_with_move(move)
			move = self.mcts.get_move(state)
			return move
		# No 'sensible' moves available, so do pass move
		return go.PASS_MOVE
