import numpy as np

WHITE = -1
BLACK = +1
EMPTY = 0

class Game(object):
	"""State of a game of Go
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
		other = Game(self.size)
		other.board = self.board.copy()
		other.turns_played = self.turns_played
		other.current_player = self.current_player
		return other

	def is_legal(self, x, y):
		"""determine if playing at (x,y) is a legal move
		"""
		empty = self.board[x][y] == EMPTY
		on_board = x >= 0 and y >= 0 and x < self.size and y < self.size
		suicide = False # todo
		ko = False # todo
		return empty and on_board and (not suicide) and (not ko)

	def move(self, x, y):
		"""Play current_player's color at (x,y)

		If it is a legal move, current_player switches to the other player
		If not, an IllegalMove exception is raised
		"""
		if self.is_legal(x,y):
			self.board[x][y] = self.current_player
			self.current_player = -self.current_player
		else:
			raise IllegalMove(str((x,y)))

	def symmetries(self):
		"""returns a list of 8 boards (size x size numpy arrays) - all
		reflections and rotations of the current board

		does not check for duplicates
		"""
		copies = [None]*8
		# the original
		copies[0] = self.board.copy()
		# rotate 90
		copies[1] = np.rot90(self.board,1)
		# rotate 180
		copies[2] = np.rot90(self.board,2)
		# rotate 270
		copies[3] = np.rot90(self.board,3)
		# mirror left-right
		copies[4] = np.fliplr(self.board)
		# mirror up-down
		copies[5] = np.flipud(self.board)
		# mirror \ diagonal
		copies[6] = np.transpose(self.board)
		# mirror / diagonal (equivalently: rotate 90 CCW then flip LR)
		copies[7] = np.fliplr(copies[1])
		return copies


class IllegalMove(Exception):
	pass