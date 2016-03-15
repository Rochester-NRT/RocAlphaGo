from AlphaGo.go import GameState
import unittest


class TestLiberties(unittest.TestCase):

	def setUp(self):
		self.s = GameState()
		self.s.do_move((4, 5))
		self.s.do_move((5, 5))
		self.s.do_move((5, 6))
		self.s.do_move((10, 10))
		self.s.do_move((4, 6))
		self.s.do_move((10, 11))
		self.s.do_move((6, 6))
		self.s.do_move((9, 10))

	def test_curr_liberties(self):
		self.assertEqual(self.s.liberty_counts[5][5], 2)
		self.assertEqual(self.s.liberty_counts[4][5], 8)
		self.assertEqual(self.s.liberty_counts[5][6], 8)

	def test_neighbors_edge_cases(self):

		st = GameState()
		st.do_move((0, 0))  # B B . . . . .
		st.do_move((5, 5))  # B W . . . . .
		st.do_move((0, 1))  # . . . . . . .
		st.do_move((6, 6))  # . . . . . . .
		st.do_move((1, 0))  # . . . . . W .
		st.do_move((1, 1))  # . . . . . . W

		# get_group in the corner
		self.assertEqual(len(st.get_group((0, 0))), 3, "group size in corner")

		# get_group of an empty space
		self.assertEqual(len(st.get_group((4, 4))), 0, "group size of empty space")

		# get_group of a single piece
		self.assertEqual(len(st.get_group((5, 5))), 1, "group size of single piece")

if __name__ == '__main__':
	unittest.main()
