import numpy as np

WHITE = -1
BLACK = +1
EMPTY = 0
PASS_MOVE = None

class GameState(object):
	"""State of a game of Go and some basic functions to interact with it
	"""

	def __init__(self, size=19):
		self.board = np.zeros((size, size))
		self.board.fill(EMPTY)
		self.size = size
		self.turns_played = 0
		self.current_player = BLACK
		self.ko = None
		self.history = []
		self.num_black_prisoners = 0
		self.num_white_prisoners = 0
	
	def liberty_count(self, position):
		"""Count liberty of a single position (maxium = 4).

		Keyword arguments:
		position -- a tuple of (x, y)
		x being the column index of the position we want to calculate the liberty
		y being the row index of the position we want to calculate the liberty

		Return:
		q -- A interger in [0, 4]. The count of liberty of the input single position
		"""
		return len(self.liberty_pos(position))

	def liberty_pos(self, position):
		"""Record the liberty position of a single position. 

		Keyword arguments:
		position -- a tuple of (x, y)
		x being the column index of the position we want to calculate the liberty
		y being the row index of the position we want to calculate the liberty

		Return:
		pos -- Return a list of tuples consist of (x, y)s which are the liberty positions on the input single position. len(pos) <= 4
		"""
		(x, y) = position
		pos=[]
		if x+1 < self.size and self.board[x+1][y] == EMPTY:
			pos.append((x+1, y))
		if y+1 < self.size and self.board[x][y+1] == EMPTY:
			pos.append((x, y+1))
		if x - 1 >= 0 and self.board[x-1][y] == EMPTY:
			pos.append((x-1, y))
		if y - 1 >= 0 and self.board[x][y-1] == EMPTY:
			pos.append((x, y-1))
		return pos

	def get_neighbor(self, position):
		"""An auxiliary function for curr_liberties. This function looks around locally in 4 directions. That is, we just pick one position and look to see if there are same-color neighbors around it. 

		Keyword arguments:
		position -- a tuple of (x, y)
		x being the column index of the position in consideration
		y being the row index of the posisiton in consideration

		Return:
		neighbor -- Return a list of tuples consist of (x, y)s which are the same-color neighbors of the input single position. len(neighbor_set) <= 4
		"""
		(x, y) = position
		neighbor_set=[]
		if y+1 < self.size and self.board[x][y] == self.board[x][y+1]:
			neighbor_set.append((x,y+1))
		if x+1 < self.size and self.board[x][y] == self.board[x+1][y]:
			neighbor_set.append((x+1,y))
		if x-1 >= 0 and self.board[x][y] == self.board[x-1][y]:
			neighbor_set.append((x-1,y))	
		if y-1 >= 0 and self.board[x][y] == self.board[x][y-1]:
			neighbor_set.append((x,y-1))	
		return neighbor_set

	def visit_neighbor(self, position):
		"""An auxiliary function for curr_liberties. This function perform the visiting process to identify a connected group of the same color

		Keyword arguments:
		position -- a tuple of (x, y)
		x being the column index of the starting position of the search
		y being the row index of the starting position of the search

		Return:
		neighbor_set -- Return a set of tuples consist of (x, y)s which are the same-color cluster which contains the input single position. len(neighbor_set) is size of the cluster, can be large. 
		"""
		(x, y) = position
		# handle case where there is no piece at (x,y)
		if self.board[x][y] == EMPTY:
			return set()
		# A list for record the places we visited in the process
		# default to the starting position to handle the case where there are no neighbors (group size is 1)
		visited=[(x,y)] 
		# A list for the the places we still want to visit
		to_visit=self.get_neighbor((x,y))
		while len(to_visit)!=0:
			for n in to_visit:
				# append serve as the actual visit
				visited.append(n)
				# take off the places already visited from the wish list
				to_visit.remove(n)
			# With the cluster we have now, we look around even further
			for v in visited:
				# we try to look for same-color neighbors for each one which we already visited
				for n in self.get_neighbor(v):
					# we don't need to consider the places we already visited when we're looking
					if n not in visited:
						to_visit.append(n)

		neighbor_set=set(visited)
		return neighbor_set

	def update_current_liberties(self):
		"""Calculate the liberty values of the whole board

		Keyword arguments:
		None. We just need the board itself.

		Return:
		A matrix self.size * self.size, with entries of the liberty number of each position on the board.
		Empty spaces have liberty 0. Instead of the single stone liberty, we consider the liberty of the
		group/cluster of the same color the position is in. 
		"""

		curr_liberties = np.ones((self.size, self.size)) * (-1)

		for x in range(0, self.size):
			for y in range(0, self.size):

				if self.board[x][y] == EMPTY:
					continue

				# get the members in the cluster and then calculate their liberty positions
				lib_set = set()
				neighbors = self.visit_neighbor((x,y))
				for n in neighbors:
					lib_set |= set(self.liberty_pos(n))
				
				curr_liberties[x][y] = len(lib_set)
		return curr_liberties

	def copy(self):
		"""get a copy of this Game state
		"""
		other = GameState(self.size)
		other.board = self.board.copy()
		other.turns_played = self.turns_played
		other.current_player = self.current_player
		other.ko = self.ko
		other.history = self.history
		other.num_black_prisoners = self.num_black_prisoners
		other.num_white_prisoners = self.num_white_prisoners
		return other

	def is_suicide(self, action):
		"""return true if having current_player play at <action> would be suicide
		"""
		tmp = self.copy()
		tmp.board[action] = tmp.current_player
		zero_liberties = tmp.update_current_liberties() == 0
		other_player = tmp.board == -tmp.current_player
		to_remove = np.logical_and(zero_liberties, other_player)
		tmp.board[to_remove] = EMPTY
		return tmp.update_current_liberties()[action] == 0

	def is_legal(self, action):
		"""determine if the given action (x,y tuple) is a legal move

		note: we only check ko, not superko at this point (TODO?)
		"""
		# passing move
		if action is PASS_MOVE:
			return True
		(x,y) = action
		empty = self.board[x][y] == EMPTY
		on_board = x >= 0 and y >= 0 and x < self.size and y < self.size
		suicide = self.is_suicide(action)
		ko = action == self.ko
		return on_board and (not suicide) and (not ko) and empty 

	def is_eye(self, position, owner):
		"""returns whether the position is empty and is surrounded by all stones of 'owner'
		"""
		(x,y) = position
		if self.board[x,y] != EMPTY:
			return False

		neighbors = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
		for (nx,ny) in neighbors:
			if nx >= 0 and ny >= 0 and nx < self.size and ny < self.size:
				if self.board[nx,ny] != owner:
					return False
		return True

	def get_legal_moves(self):
		moves = []
		for x in range(self.size):
			for y in range(self.size):
				if self.is_legal((x,y)):
					moves.append((x,y))
		return moves

	def do_move(self, action):
		"""Play current_player's color at (x,y)

		If it is a legal move, current_player switches to the other player
		If not, an IllegalMove exception is raised
		"""
		if self.is_legal(action):
			# reset ko
			self.ko = None
			if action is not PASS_MOVE:
				(x,y) = action
				self.board[x][y] = self.current_player
				
				# check liberties for captures
				liberties = self.update_current_liberties()
				zero_liberties = liberties == 0
				other_player = self.board == -self.current_player
				captured_stones = np.logical_and(zero_liberties, other_player)
				capture_occurred = np.any(captured_stones) # note EMPTY spaces are -1
				if capture_occurred:
					# clear pieces
					self.board[captured_stones] = EMPTY
					# count prisoners
					num_captured = np.sum(captured_stones)
					if self.current_player == BLACK:
						self.num_white_prisoners += num_captured
					else:
						self.num_black_prisoners += num_captured
					if num_captured == 1:
						xcoord,ycoord = np.where(captured_stones)
						# it is a ko iff were the opponent to play at xcoord,ycoord
						# it would recapture (x,y) only
						# (a bigger group containing xy may be captured - this is 'snapback')
						if len(self.visit_neighbor(action)) == 1 and self.update_current_liberties()[action] == 1:
							self.ko = (xcoord[0], ycoord[0])
			# next turn
			self.current_player = -self.current_player
			self.turns_played += 1
			self.history.append(action)
		else:
			raise IllegalMove(str(action))

	def symmetries(self):
		"""returns a list of 8 GameState objects:
		all reflections and rotations of the current board

		does not check for duplicates
		"""

		# we use numpy's built-in array symmetry routines for self.board.
		# but for all xy pairs (i.e. self.ko and self.history), we need to
		# know how to rotate a tuple (x,y) into (new_x, new_y)
		xy_symmetry_functions = {
			"noop":   lambda (x,y): (x, y),
			"rot90":  lambda (x,y): (y, self.size-x),
			"rot180": lambda (x,y): (self.size-x, self.size-y),
			"rot270": lambda (x,y): (self.size-y, x),
			"mirror-lr": lambda (x,y): (self.size-x, y),
			"mirror-ud": lambda (x,y): (x, self.size-y),
			"mirror-\\": lambda (x,y): (y, x),
			"mirror-/":  lambda (x,y): (self.size-y, self.size-x)
		}

		def update_ko_history(copy, name):
			if copy.ko is not None:
				copy.ko = xy_symmetry_functions[name](copy.ko)
			copy.history = [xy_symmetry_functions[name](a) if a is not PASS_MOVE else PASS_MOVE for a in copy.history]

		copies = [self.copy() for i in range(8)]
		# copies[0] is the original.
		# rotate CCW 90
		copies[1].board = np.rot90(self.board,1)
		update_ko_history(copies[1], "rot90")
		# rotate 180
		copies[2].board = np.rot90(self.board,2)
		update_ko_history(copies[2], "rot180")
		# rotate CCW 270
		copies[3].board = np.rot90(self.board,3)
		update_ko_history(copies[3], "rot270")
		# mirror left-right
		copies[4].board = np.fliplr(self.board)
		update_ko_history(copies[4], "mirror-lr")
		# mirror up-down
		copies[5].board = np.flipud(self.board)
		update_ko_history(copies[5], "mirror-ud")
		# mirror \ diagonal
		copies[6].board = np.transpose(self.board)
		update_ko_history(copies[6], "mirror-\\")
		# mirror / diagonal (equivalently: rotate 90 CCW then flip LR)
		copies[7].board = np.fliplr(copies[1].board)
		update_ko_history(copies[7], "mirror-/")
		return copies

	def from_sgf(self, sgf_string):
		raise NotImplementedError()

	def to_sgf(self, sgf_string):
		raise NotImplementedError()


class IllegalMove(Exception):
	pass