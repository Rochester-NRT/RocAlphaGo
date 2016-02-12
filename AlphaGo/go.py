import numpy as np
from sgflib.sgflib import SGFParser

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

	@classmethod
	def from_sgf(cls, sgf):
		"""returns a generator of GameState objects read from the given sgf string

		the sgf format allows specifying multiple GameTrees per file. Each GameTree
		is converted into a GameState object
		"""
		def alphabet_to_xy(ab):
			"""convert from alphabet notation of column/row to (x,y) pair

			for example, alphabet_to_xy('ei') returns (4,8)
			"""
			ab = ab.lower()
			x = ord(ab[0]) - ord('a')
			y = ord(ab[1]) - ord('a')
			return (x,y)

		def gametree_to_gamestate(gt):
			"""convert from sgflib format to a GameState object
			"""
			gs = GameState()

			# loop over and replay each move
			i = 0
			for node in gt:
				if node.has_key('B'):
					print 'B', i, node['B'][0]
					gs.current_player = BLACK
					gs.do_move(alphabet_to_xy(node['B'][0]))
				if node.has_key('W'):
					print 'W', i, node['W'][0]
					gs.current_player = WHITE
					gs.do_move(alphabet_to_xy(node['W'][0]))
				i = i+1
			return gs

		parser = SGFParser(sgf)
		gametree = parser.parseOneGame()
		while gametree is not None:
			# convert from sgflib format to a GameState object
			yield gametree_to_gamestate(gametree)
			# prepare next while loop iteration
			gametree = parser.parseOneGame()

	def to_sgf(self):
		raise NotImplementedError()


class IllegalMove(Exception):
	pass