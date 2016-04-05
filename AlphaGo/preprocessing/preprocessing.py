import numpy as np
import AlphaGo.go as go

##
## individual feature functions (state --> tensor) begin here
##


def get_board(state):
	"""A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
	always refers to the current player and plane 1 to the opponent
	"""
	planes = np.zeros((state.size, state.size, 3))
	planes[:, :, 0] = state.board == state.current_player  # own stone
	planes[:, :, 1] = state.board == -state.current_player  # opponent stone
	planes[:, :, 2] = state.board == go.EMPTY  # empty space
	return planes


def get_turns_since(state, maximum=8):
	"""A feature encoding the age of the stone at each location up to 'maximum'

	Note:
	- the [maximum-1] plane is used for any stone with age greater than or equal to maximum
	- EMPTY locations are all-zero features
	"""
	planes = np.zeros((state.size, state.size, maximum))
	depth = 0
	# loop backwards over history and place a 1 in plane 0
	# for the most recent move, a 1 in plane 1 for two moves ago, etc..
	for move in state.history[::-1]:
		if move is not go.PASS_MOVE:
			# check that this stone wasn't captured
			if state.board[move] != go.EMPTY:
				(x, y) = move
				# check that a newer move isn't occupying (x,y)
				if np.sum(planes[x, y, :]) == 0:
					planes[x, y, depth] = 1
		# increment depth if there are more planes available
		# (the last plane serves as the "maximum-1 or more" feature)
		if depth < maximum - 1:
			depth += 1
	return planes


def get_liberties(state, maximum=8):
	"""A feature encoding the number of liberties of the group connected to the stone at
	each location

	Note:
	- there is no zero-liberties plane; the 0th plane indicates groups in atari
	- the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
	- EMPTY locations are all-zero features
	"""
	planes = np.zeros((state.size, state.size, maximum))
	for i in range(maximum):
		# single liberties in plane zero (groups won't have zero), double liberties in plane one, etc
		planes[state.liberty_counts == i + 1, i] = 1
	# the "maximum-or-more" case on the backmost plane
	planes[state.liberty_counts >= maximum, maximum - 1] = 1
	return planes


def get_capture_size(state, maximum=8):
	"""A feature encoding the number of opponent stones that would be captured by planing at each location,
	up to 'maximum'

	Note:
	- we currently *do* treat the 0th plane as "capturing zero stones"
	- the [maximum-1] plane is used for any capturable group of size greater than or equal to maximum-1
	- the 0th plane is used for legal moves that would not result in capture
	- illegal move locations are all-zero features
	"""
	planes = np.zeros((state.size, state.size, maximum))
	for (x, y) in state.get_legal_moves():
		# multiple disconnected groups may be captured. hence we loop over
		# groups and count sizes if captured.
		n_captured = 0
		for neighbor_group in state.get_groups_around((x, y)):
			# if the neighboring group is opponent stones and they have
			# one liberty, it must be (x,y) and we are capturing them
			# (note suicide and ko are not an issue because they are not
			# legal moves)
			(gx, gy) = next(iter(neighbor_group))
			if (state.liberty_counts[gx][gy] == 1) and (state.board[gx, gy] != state.current_player):
				n_captured += len(state.group_sets[gx][gy])
		planes[x, y, min(n_captured, maximum - 1)] = 1
	return planes


def get_self_atari_size(state, maximum=8):
	"""A feature encoding the size of the own-stone group that is put into atari by playing at a location
	"""
	planes = np.zeros((state.size, state.size, maximum))

	for (x, y) in state.get_legal_moves():
		# make a copy of the liberty/group sets at (x,y) so we can manipulate them
		lib_set_after = set(state.liberty_sets[x][y])
		group_set_after = set()
		group_set_after.add((x, y))
		for neighbor_group in state.get_groups_around((x, y)):
			# if the neighboring group is of the same color as the current player
			# then playing here will connect this stone to that group
			(gx, gy) = next(iter(neighbor_group))
			if state.board[gx, gy] == state.current_player:
				lib_set_after |= state.liberty_sets[gx][gy]
				group_set_after |= state.liberty_sets[gx][gy]
		if (x, y) in lib_set_after:
			lib_set_after.remove((x, y))
		# check if this move resulted in atari
		if len(lib_set_after) == 1:
			group_size = len(group_set_after)
			# 0th plane used for size=1, so group_size-1 is the index
			planes[x, y, min(group_size - 1, maximum - 1)] = 1
	return planes


def get_liberties_after(state, maximum=8):
	"""A feature encoding what the number of liberties *would be* of the group connected to
	the stone *if* played at a location

	Note:
	- there is no zero-liberties plane; the 0th plane indicates groups in atari
	- the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
	- illegal move locations are all-zero features
	"""
	feature = np.zeros((state.size, state.size, maximum))
	# note - left as all zeros if not a legal move
	for (x, y) in state.get_legal_moves():
		# make a copy of the set of liberties at (x,y) so we can add to it
		lib_set_after = set(state.liberty_sets[x][y])
		for neighbor_group in state.get_groups_around((x, y)):
			# if the neighboring group is of the same color as the current player
			# then playing here will connect this stone to that group and
			# therefore add in all that group's liberties
			(gx, gy) = next(iter(neighbor_group))
			if state.board[gx, gy] == state.current_player:
				lib_set_after |= state.liberty_sets[gx][gy]
		# (x,y) itself may have made its way back in, but shouldn't count
		# since it's clearly not a liberty after playing there
		if (x, y) in lib_set_after:
			lib_set_after.remove((x, y))
		feature[x, y, min(maximum - 1, len(lib_set_after) - 1)] = 1
	return feature


def get_ladder_capture(state):
	raise NotImplementedError()


def get_ladder_escape(state):
	raise NotImplementedError()


def get_sensibleness(state):
	"""A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
	"""
	feature = np.zeros((state.size, state.size))
	for (x, y) in state.get_legal_moves():
		if not state.is_eye((x, y), state.current_player):
			feature[x, y] = 1
	return feature

# named features and their sizes are defined here
FEATURES = {
	"board": {
		"size": 3,
		"function": get_board
	},
	"ones": {
		"size": 1,
		"function": lambda state: np.ones((state.size, state.size))
	},
	"turns_since": {
		"size": 8,
		"function": get_turns_since
	},
	"liberties": {
		"size": 8,
		"function": get_liberties
	},
	"capture_size": {
		"size": 8,
		"function": get_capture_size
	},
	"self_atari_size": {
		"size": 8,
		"function": get_self_atari_size
	},
	"liberties_after": {
		"size": 8,
		"function": get_liberties_after
	},
	"ladder_capture": {
		"size": 1,
		"function": get_ladder_capture
	},
	"ladder_escape": {
		"size": 1,
		"function": get_ladder_escape
	},
	"sensibleness": {
		"size": 1,
		"function": get_sensibleness
	},
	"zeros": {
		"size": 1,
		"function": lambda state: np.zeros((state.size, state.size))
	}
}

DEFAULT_FEATURES = [
	"board", "ones", "turns_since", "liberties", "capture_size",
	"self_atari_size", "liberties_after", "ladder_capture", "ladder_escape",
	"sensibleness", "zeros"]


class Preprocess(object):
	"""a class to convert from AlphaGo GameState objects to tensors of one-hot
	features for NN inputs
	"""

	def __init__(self, feature_list=DEFAULT_FEATURES):
		"""create a preprocessor object that will concatenate together the
		given list of features
		"""

		self.output_dim = 0
		self.feature_list = feature_list
		self.processors = [None] * len(feature_list)
		for i in range(len(feature_list)):
			feat = feature_list[i].lower()
			if feat in FEATURES:
				self.processors[i] = FEATURES[feat]["function"]
				self.output_dim += FEATURES[feat]["size"]
			else:
				raise ValueError("uknown feature: %s" % feat)

	def state_to_tensor(self, state):
		"""Convert a GameState to a Theano-compatible tensor
		"""
		feat_tensors = [proc(state) for proc in self.processors]

		# TODO - make features smarter so they don't have to be transposed and reshaped,
		# just stacked, and this loop could be avoided
		for i, feat in enumerate(feat_tensors):
			# reshape (width,height,depth) to (depth,width,height)
			if feat.ndim == 2:
				(w, h) = feat.shape
				d = 1
			else:
				(w, h, d) = feat.shape
			feat_tensors[i] = feat.reshape((w, h, d)).transpose((2, 0, 1))

		# concatenate along feature dimension then add in a singleton 'batch' dimensino
		f, s = self.output_dim, state.size
		return np.concatenate(feat_tensors).reshape((1, f, s, s))
