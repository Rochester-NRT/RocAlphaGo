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
		self.s.do_move((10,11))
		self.s.do_move((6,6))
		self.s.do_move((9, 10))

		self.syms = self.s.symmetries()

	def test_lib_count(self):
		self.assertEqual(self.s.liberty_count((5,5)), 2)
		print("liberty_count checked")

	def test_lib_pos(self):
		self.assertEqual(self.s.liberty_pos((5,5)), [(6,5), (5,4)])
		print("liberty_pos checked")

	def test_curr_liberties(self):
		self.assertEqual(self.s.update_current_liberties()[5][5], 2)
		self.assertEqual(self.s.update_current_liberties()[4][5], 8)
		self.assertEqual(self.s.update_current_liberties()[5][6], 8)

		print("curr_liberties checked")

	def test_neighbors_edge_cases(self):

		st = GameState()
		st.do_move((0,0)) #  B B . . . . . 
		st.do_move((5,5)) #  B W . . . . . 
		st.do_move((0,1)) #  . . . . . . . 
		st.do_move((6,6)) #  . . . . . . . 
		st.do_move((1,0)) #  . . . . . W . 
		st.do_move((1,1)) #  . . . . . . W

		# visit_neighbor in the corner
		self.assertEqual(len(st.visit_neighbor((0,0))), 3, "group size in corner")

		# visit_neighbor of an empty space
		self.assertEqual(len(st.visit_neighbor((4,4))), 0, "group size of empty space")

		# visit_neighbor of a single piece
		self.assertEqual(len(st.visit_neighbor((5,5))), 1, "group size of single piece")