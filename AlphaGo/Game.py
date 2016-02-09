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
		return empty and on_board and (not suicide) and (not ko)

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