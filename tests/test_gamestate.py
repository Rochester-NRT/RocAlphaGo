from AlphaGo.go import GameState
import AlphaGo.go as go
import unittest


class TestKo(unittest.TestCase):

	def test_standard_ko(self):
		gs = GameState(size=9)
		gs.do_move((1, 0))  # B
		gs.do_move((2, 0))  # W
		gs.do_move((0, 1))  # B
		gs.do_move((3, 1))  # W
		gs.do_move((1, 2))  # B
		gs.do_move((2, 2))  # W
		gs.do_move((2, 1))  # B

		gs.do_move((1, 1))  # W trigger capture and ko

		self.assertEqual(gs.num_black_prisoners, 1)
		self.assertEqual(gs.num_white_prisoners, 0)

		self.assertFalse(gs.is_legal((2, 1)))

		gs.do_move((5, 5))
		gs.do_move((5, 6))

		self.assertTrue(gs.is_legal((2, 1)))

	def test_snapback_is_not_ko(self):
		gs = GameState(size=5)
		# B o W B .
		# W W B . .
		# . . . . .
		# . . . . .
		# . . . . .
		# here, imagine black plays at 'o' capturing
		# the white stone at (2,0). White may play
		# again at (2,0) to capture the black stones
		# at (0,0), (1,0). this is 'snapback' not 'ko'
		# since it doesn't return the game to a
		# previous position
		B = [(0, 0), (2, 1), (3, 0)]
		W = [(0, 1), (1, 1), (2, 0)]
		for (b, w) in zip(B, W):
			gs.do_move(b)
			gs.do_move(w)
		# do the capture of the single white stone
		gs.do_move((1, 0))
		# there should be no ko
		self.assertIsNone(gs.ko)
		self.assertTrue(gs.is_legal((2, 0)))
		# now play the snapback
		gs.do_move((2, 0))
		# check that the numbers worked out
		self.assertEqual(gs.num_black_prisoners, 2)
		self.assertEqual(gs.num_white_prisoners, 1)


class TestEye(unittest.TestCase):

	def test_simple_eye(self):

		# create a black eye in top left (1,1), white in bottom right (5,5)

		gs = GameState(size=7)
		gs.do_move((1, 0))  # B
		gs.do_move((5, 4))  # W
		gs.do_move((2, 1))  # B
		gs.do_move((6, 5))  # W
		gs.do_move((1, 2))  # B
		gs.do_move((5, 6))  # W
		gs.do_move((0, 1))  # B
		gs.do_move((4, 5))  # W

		# test black eye top left
		self.assertTrue(gs.is_eye((1, 1), go.BLACK))
		self.assertFalse(gs.is_eye((1, 1), go.WHITE))

		# test white eye bottom right
		self.assertTrue(gs.is_eye((5, 5), go.WHITE))
		self.assertFalse(gs.is_eye((5, 5), go.BLACK))

		# test no eye in other random positions
		self.assertFalse(gs.is_eye((1, 0), go.BLACK))
		self.assertFalse(gs.is_eye((1, 0), go.WHITE))
		self.assertFalse(gs.is_eye((2, 2), go.BLACK))
		self.assertFalse(gs.is_eye((2, 2), go.WHITE))

if __name__ == '__main__':
	unittest.main()
