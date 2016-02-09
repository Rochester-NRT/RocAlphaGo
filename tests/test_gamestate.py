from AlphaGo.Game import GameState
import numpy as np
import unittest

class TestSymmetries(unittest.TestCase):

	def setUp(self):
		self.s = GameState()
		self.s.do_move((4,5))
		self.s.do_move((5,5))
		self.s.do_move((5,6))

		self.syms = self.s.symmetries()

	def test_num_syms(self):
		# make sure we got exactly 8 back
		self.assertEqual(len(self.syms), 8)

	def test_copy_fields(self):
		# make sure each copy has the correct non-board fields
		for copy in self.syms:
			self.assertEqual(self.s.size, copy.size)
			self.assertEqual(self.s.turns_played, copy.turns_played)
			self.assertEqual(self.s.current_player, copy.current_player)

	def test_sym_boards(self):
		# construct by hand the 8 boards we expect to see
		expectations = [GameState() for i in range(8)]

		descriptions = ["noop", "rot90", "rot180", "rot270", "mirror LR", "mirror UD", "mirror \\", "mirror /"]

		# copy of self.s
		expectations[0].do_move((4,5))
		expectations[0].do_move((5,5))
		expectations[0].do_move((5,6))

		# rotate 90 CCW
		expectations[1].do_move((13,4))
		expectations[1].do_move((13,5))
		expectations[1].do_move((12,5))

		# rotate 180
		expectations[2].do_move((14,13))
		expectations[2].do_move((13,13))
		expectations[2].do_move((13,12))

		# rotate CCW 270
		expectations[3].do_move((5,14))
		expectations[3].do_move((5,13))
		expectations[3].do_move((6,13))

		# mirror left-right
		expectations[4].do_move((4,13))
		expectations[4].do_move((5,13))
		expectations[4].do_move((5,12))

		# mirror up-down
		expectations[5].do_move((14,5))
		expectations[5].do_move((13,5))
		expectations[5].do_move((13,6))

		# mirror \ diagonal
		expectations[6].do_move((5,4))
		expectations[6].do_move((5,5))
		expectations[6].do_move((6,5))

		# mirror / diagonal (equivalently: rotate 90 CCW then flip LR)
		expectations[7].do_move((13,14))
		expectations[7].do_move((13,13))
		expectations[7].do_move((12,13))

		for i in range(8):
			self.assertTrue(np.array_equal(expectations[i].board, self.syms[i].board), descriptions[i])