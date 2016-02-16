import numpy as np

WHITE = -1
BLACK = +1
EMPTY = 0

class GameState(object):
	"""State of a game of Go and some basic functions to interact with it
	"""

	def __init__(self, size=19):
		self.board = np.zeros((size, size))
		self.board.fill(EMPTY)
		self.size = size
		self.turns_played = 0
		self.current_player = BLACK
	
	def liberty_count(self, position):
		"""Count liberty of a single position (maxium = 4).

	    Keyword arguments:
	    position -- a tuple of (x, y)
	    x being the row index of the position we want to calculate the liberty
	    y being the column index of the position we want to calculate the liberty

	    Return:
	    q -- A interger in [0, 4]. The count of liberty of the input single position
	    """
		(x, y) = position
		q=0 #liberty count
		if x+1 < self.size and self.board[x+1][y] == EMPTY:
			q = q + 1
		if y+1 < self.size and self.board[x][y+1] == EMPTY:
			q = q + 1
		if x-1 > 0 and self.board[x-1][y] == EMPTY:
			q = q + 1
		if y -1 > 0 and self.board[x][y -1] == EMPTY:
			q = q + 1
		return q

	def liberty_pos(self, position):
		"""Record the liberty position of a single position. 

		Keyword arguments:
		position -- a tuple of (x, y)
	    x being the row index of the position we want to calculate the liberty
	    y being the column index of the position we want to calculate the liberty

	    Return:
	    pos -- Return a list of tuples consist of (x, y)s which are the liberty positions on the input single position. len(tuple(pos)) <= 4
		"""
		(x, y) = position
		pos=[]
		if x+1<self.size and self.board[x+1][y] == EMPTY:
			pos.append(tuple([x+1, y]))
		if y+1<self.size and self.board[x][y+1] == EMPTY:
			pos.append(tuple([x, y+1]))
		if x - 1 >= 0 and self.board[x-1][y] == EMPTY:
			pos.append(tuple([x-1, y]))
		if y - 1 >= 0 and self.board[x][y-1] == EMPTY:
			pos.append(tuple([x, y-1]))
		return pos

	def get_neighbor(self, (x, y)):
		"""An auxiliary function for curr_liberties. This function looks around locally in 4 directions. That is, we just pick one position and look to see if there are same-color neighbors around it. 

		Keyword arguments:
		position -- a tuple of (x, y)
	    x being the row index of the position in consideration
	    y being the column index of the posisiton in consideration

	    Return:
	    neighbor -- Return a list of tuples consist of (x, y)s which are the same-color neighbors of the input single position. len(neighbor_set) <= 4
		"""
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

	def visit_neighbor(self, (x,y)):
		"""An auxiliary function for curr_liberties. This function perform the visiting process when we try to identify the cluster of the same color

		Keyword arguments:
		position -- a tuple of (x, y)
	    x being the row index of the starting position of the search
	    y being the column index of the starting position of the search

	    Return:
	    neighbor_set -- Return a set of tuples consist of (x, y)s which are the same-color cluster which contains the input single position. len(neighbor_set) is size of the cluster, can be large. 
		"""
		# A list for record the places we visited in the process
		visited=[] 
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
	    A matrix self.size * self.size, with entries of the liberty number of each position on the board. Empty spaces have liberty -1. Instead of the single stone liberty, we consider the liberty of the group/cluster of the same color the position is in. 
	    """

		curr_liberties=np.ones((self.size, self.size))*(-1)

		for x in range(0, self.size):
			for y in range(0, self.size):

				if self.board[x][y] == EMPTY:
					continue

				# The first position picked
				lib_set = []
				for p in self.liberty_pos((x, y)):
						lib_set.append(p)

				# Using 2 auxiliary functions to get the members in the cluster and then calculate their liberty position
				neighbors=self.visit_neighbor((x,y))
				for n in neighbors:
					poslist=self.liberty_pos(n)
					for p in poslist:
						lib_set.append(p)
				
				# Combine the liberty position of the cluster found
				lib_set = set(lib_set)
				curr_liberties[x][y] = len(lib_set)
		return curr_liberties

	def update_future_liberties(self, action):
		"""Calculate the liberty values of the whole board after we make a new move

		Keyword arguments:
		action -- a tuple of (x, y)
	    x being the column index of the position of the future move
	    y being the row index of the position of the future move

	    Return:
	    A matrix self.size * self.size, with entries of the liberty number of each position on the board, after the future move. 
	    """ 
		future = self.copy()
		future.do_move(action)
		future_liberties = future.update_current_liberties()

		return future_liberties

	def copy(self):
		"""get a copy of this Game state
		"""
		other = GameState(self.size)
		other.board = self.board.copy()
		other.turns_played = self.turns_played
		other.current_player = self.current_player
		return other

	def is_legal(self, action):
		"""determine if the given action (x,y tuple) is a legal move
		"""
		(x,y) = action
		empty = self.board[x][y] == EMPTY
		on_board = x >= 0 and y >= 0 and x < self.size and y < self.size
		suicide = False # todo
		ko = False # todo
		return on_board and (not suicide) and (not ko) #and empty 

	def do_move(self, action):
		"""Play current_player's color at (x,y)

		If it is a legal move, current_player switches to the other player
		If not, an IllegalMove exception is raised
		"""
		(x,y) = action
		if self.is_legal((x,y)):
			self.board[x][y] = self.current_player
			self.current_player = -self.current_player
			self.turns_played += 1
		else:
			raise IllegalMove(str((x,y)))

	def symmetries(self):
		"""returns a list of 8 GameState objects:
		all reflections and rotations of the current board

		does not check for duplicates
		"""
		copies = [self.copy() for i in range(8)]
		# copies[0] is the original.
		# rotate CCW 90
		copies[1].board = np.rot90(self.board,1)
		# rotate 180
		copies[2].board = np.rot90(self.board,2)
		# rotate CCW 270
		copies[3].board = np.rot90(self.board,3)
		# mirror left-right
		copies[4].board = np.fliplr(self.board)
		# mirror up-down
		copies[5].board = np.flipud(self.board)
		# mirror \ diagonal
		copies[6].board = np.transpose(self.board)
		# mirror / diagonal (equivalently: rotate 90 CCW then flip LR)
		copies[7].board = np.fliplr(copies[1].board)
		return copies

	def from_sgf(self, sgf_string):
		raise NotImplementedError()

	def to_sgf(self, sgf_string):
		raise NotImplementedError()


class IllegalMove(Exception):
	pass