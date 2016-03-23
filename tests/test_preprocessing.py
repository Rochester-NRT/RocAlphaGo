from AlphaGo.preprocessing.preprocessing import Preprocess
import AlphaGo.go as go
import numpy as np
import unittest

def simple_board():
	# make a tiny board for the sake of testing and hand-coding expected results
	#
	#         X
	#   0 1 2 3 4 5 6
	#   B W . . . . . 0
	#   B W . . . . . 1
	#   B . . . B . . 2
	# Y . . . B k B . 3
	#   . . . W B W . 4
	#   . . . . W . . 5
	#   . . . . . . . 6
	#
	# where k is a ko position (white was just captured)

	gs = go.GameState(size=7)

	# ladder-looking thing in the top-left
	gs.do_move((0,0)) # B
	gs.do_move((1,0)) # W
	gs.do_move((0,1)) # B
	gs.do_move((1,1)) # W
	gs.do_move((0,2)) # B

	# ko position in the middle
	gs.do_move((3,4)) # W
	gs.do_move((3,3)) # B
	gs.do_move((4,5)) # W
	gs.do_move((4,2)) # B
	gs.do_move((5,4)) # W
	gs.do_move((5,3)) # B
	gs.do_move((4,3)) # W - the ko position
	gs.do_move((4,4)) # B - does the capture

	return gs

class TestPreprocessingFeatures(unittest.TestCase):
	"""Test the functions in preprocessing.py

	note that the hand-coded features look backwards from what is depicted
	in simple_board() because of the x/y column/row transpose thing (i.e.
	numpy is typically thought of as indexing rows first, but we use (x,y)
	indexes, so a numpy row is like a go column and vice versa)
	"""


	def test_get_board(self):
		gs = simple_board()
		pp = Preprocess(["board"])
		feature = pp.state_to_tensor(gs)[0].transpose((1,2,0))

		white_pos = np.asarray([
			[0,0,0,0,0,0,0],
			[1,1,0,0,0,0,0],
			[0,0,0,0,0,0,0],
			[0,0,0,0,1,0,0],
			[0,0,0,0,0,1,0],
			[0,0,0,0,1,0,0],
			[0,0,0,0,0,0,0]])
		black_pos = np.asarray([
			[1,1,1,0,0,0,0],
			[0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0],
			[0,0,0,1,0,0,0],
			[0,0,1,0,1,0,0],
			[0,0,0,1,0,0,0],
			[0,0,0,0,0,0,0]])
		empty_pos = np.ones((gs.size, gs.size)) - (white_pos + black_pos)

		# check number of planes
		self.assertEqual(feature.shape, (gs.size, gs.size, 3))
		# check return value against hand-coded expectation
		# (given that current_player is white)
		self.assertTrue(np.all(feature == np.dstack((white_pos, black_pos, empty_pos))))

	def test_get_turns_since(self):
		gs = simple_board()
		pp = Preprocess(["turns_since"])
		feature = pp.state_to_tensor(gs)[0].transpose((1,2,0))

		one_hot_turns = np.zeros((gs.size, gs.size, 8))
		rev_history = gs.history[::-1]
		# one plane per move for the last 7
		for i in range(7):
			move = rev_history[i]
			one_hot_turns[move[0], move[1], i] = 1
		# far back plane gets all other moves
		for move in rev_history[7:]:
			one_hot_turns[move[0], move[1], 7] = 1

		self.assertTrue(np.all(feature == one_hot_turns))

	def test_get_liberties(self):
		gs = simple_board()
		pp = Preprocess(["liberties"])
		feature = pp.state_to_tensor(gs)[0].transpose((1,2,0))

		# todo - test liberties when > 8

		one_hot_liberties = np.zeros((gs.size, gs.size, 8))
		# black piece at (4,4) has a single liberty: (4,3)
		one_hot_liberties[4,4,0] = 1

		# the black group in the top left corner has 2 liberties
		one_hot_liberties[0,0:3,1] = 1
		# 	.. as do the white pieces on the left and right of the eye
		one_hot_liberties[3,4,1] = 1
		one_hot_liberties[5,4,1] = 1

		# the white group in the top left corner has 3 liberties
		one_hot_liberties[1,0:2,2] = 1
		# 	...as does the white piece at (4,5)
		one_hot_liberties[4,5,2] = 1
		# 	...and the black pieces on the sides of the eye
		one_hot_liberties[3,3,2] = 1
		one_hot_liberties[5,3,2] = 1

		# the black piece at (4,2) has 4 liberties
		one_hot_liberties[4,2,3] = 1

		for i in range(8):
			self.assertTrue(np.all(feature[:,:,i] == one_hot_liberties[:,:,i]),
				"bad expectation: stones with %d liberties" % (i+1))

	def test_get_capture_size(self):
		# TODO - at the moment there is no imminent capture
		gs = simple_board()
		pp = Preprocess(["capture_size"])
		feature = pp.state_to_tensor(gs)[0].transpose((1,2,0))

		one_hot_capture = np.zeros((gs.size, gs.size, 8))
		# there is no capture available; all legal moves are zero-capture
		for (x,y) in gs.get_legal_moves():
			one_hot_capture[x,y,0] = 1

		for i in range(8):
			self.assertTrue(np.all(feature[:,:,i] == one_hot_capture[:,:,i]),
				"bad expectation: capturing %d stones" % i)

	def test_get_self_atari_size(self):
		# TODO - at the moment there is no imminent self-atari for white
		gs = simple_board()
		pp = Preprocess(["self_atari_size"])
		feature = pp.state_to_tensor(gs)[0].transpose((1,2,0))

		self.assertTrue(np.all(feature == np.zeros((gs.size, gs.size, 8))))

	def test_get_liberties_after(self):
		gs = simple_board()
		pp = Preprocess(["liberties_after"])
		feature = pp.state_to_tensor(gs)[0].transpose((1,2,0))

		one_hot_liberties = np.zeros((gs.size, gs.size, 8))

		# TODO (?) hand-code?
		for (x,y) in gs.get_legal_moves():
			copy = gs.copy()
			copy.do_move((x,y))
			libs = copy.liberty_counts[x,y]
			if libs < 7:
				one_hot_liberties[x,y,libs] = 1
			else:
				one_hot_liberties[x,y,7] = 1

		for i in range(8):
			self.assertTrue(np.all(feature[:,:,i] == one_hot_liberties[:,:,i]),
				"bad expectation: stones with %d liberties after move" % (i+1))

	def test_get_ladder_capture(self):
		pass

	def test_get_ladder_escape(self):
		pass

	def test_get_sensibleness(self):
		# TODO - there are no legal eyes at the moment

		gs = simple_board()
		pp = Preprocess(["sensibleness"])
		feature = pp.state_to_tensor(gs)[0,0] # 1D tensor; no need to transpose

		expectation = np.zeros((gs.size, gs.size))
		for (x,y) in gs.get_legal_moves():
			if not (gs.is_eye((x,y), go.WHITE)):
				expectation[x,y] = 1
		self.assertTrue(np.all(expectation == feature))

	def test_feature_concatenation(self):
		gs = simple_board()
		pp = Preprocess(["board", "sensibleness", "capture_size"])
		feature = pp.state_to_tensor(gs)[0].transpose((1,2,0))

		expectation = np.zeros((gs.size, gs.size, 3+1+8))

		# first three planes: board
		expectation[:,:,0] = (gs.board == go.WHITE) * 1
		expectation[:,:,1] = (gs.board == go.BLACK) * 1
		expectation[:,:,2] = (gs.board == go.EMPTY) * 1

		# 4th plane: sensibleness (as in test_get_sensibleness)
		for (x,y) in gs.get_legal_moves():
			if not (gs.is_eye((x,y), go.WHITE)):
				expectation[x,y,3] = 1

		# 5th through 12th plane: capture size (all zero-capture)
		for (x,y) in gs.get_legal_moves():
			expectation[x,y,4] = 1

		self.assertTrue(np.all(expectation == feature))

if __name__ == '__main__':
	unittest.main()
