from AlphaGo.go import GameState
import numpy as np
import unittest

class TestLiberties(unittest.TestCase):

	def setUp(self):
		self.s = GameState()
		self.s.do_move((4,5))
		self.s.do_move((5,5))
		self.s.do_move((5,6))
		self.s.do_move((10,10))
		self.s.do_move((4,6))

		self.syms = self.s.symmetries()

	def test_lib_count(self):
		self.assertEqual(self.s.liberty_count(5,5), 2)
		print("liberty_count checked")

	def test_lib_pos(self):
		self.assertEqual(self.s.liberty_pos(5,5), ((6,5), (5,4)))
		print("liberty_pos checked")

	def test_curr_liberties(self):
		self.assertEqual(self.s.update_current_liberties()[5][5], 2)
		self.assertEqual(self.s.update_current_liberties()[4][5], 6)
		self.assertEqual(self.s.update_current_liberties()[5][6], 6)

		print("curr_liberties checked")

	def test_future_liberties(self):

		self.assertEqual(self.s.update_future_liberties((6,5))[6][5], 4)
		self.assertEqual(self.s.update_future_liberties((5,4))[5][4], 4)
		self.assertEqual(self.s.update_future_liberties((6,6))[5][6], 5)

		print("future_liberties checked")
