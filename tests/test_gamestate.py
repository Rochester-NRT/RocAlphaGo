from AlphaGo.go import GameState
import numpy as np
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
        # the white stone at (2, 0). White may play
        # again at (2, 0) to capture the black stones
        # at (0, 0), (1, 0). this is 'snapback' not 'ko'
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

    def test_positional_superko(self):
        move_list = [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (2, 2), (3, 4), (2, 1), (3, 3),
                     (3, 1), (3, 2), (3, 0), (4, 2), (1, 1), (4, 1), (8, 0), (4, 0), (8, 1), (0, 2),
                     (8, 2), (0, 1), (8, 3), (1, 0), (8, 4), (2, 0), (0, 0)]

        gs = GameState(size=9)
        for move in move_list:
            gs.do_move(move)
        self.assertTrue(gs.is_legal((1, 0)))

        gs = GameState(size=9, enforce_superko=True)
        for move in move_list:
            gs.do_move(move)
        self.assertFalse(gs.is_legal((1, 0)))


class TestEye(unittest.TestCase):

    def test_simple_eye(self):

        # create a black eye in top left (1, 1), white in bottom right (5, 5)

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
        self.assertTrue(gs.is_eyeish((1, 1), go.BLACK))
        self.assertFalse(gs.is_eyeish((1, 1), go.WHITE))

        # test white eye bottom right
        self.assertTrue(gs.is_eyeish((5, 5), go.WHITE))
        self.assertFalse(gs.is_eyeish((5, 5), go.BLACK))

        # test no eye in other random positions
        self.assertFalse(gs.is_eyeish((1, 0), go.BLACK))
        self.assertFalse(gs.is_eyeish((1, 0), go.WHITE))
        self.assertFalse(gs.is_eyeish((2, 2), go.BLACK))
        self.assertFalse(gs.is_eyeish((2, 2), go.WHITE))

    def test_true_eye(self):
        gs = GameState(size=7)
        gs.do_move((1, 0), go.BLACK)
        gs.do_move((0, 1), go.BLACK)

        # false eye at 0, 0
        self.assertTrue(gs.is_eyeish((0, 0), go.BLACK))
        self.assertFalse(gs.is_eye((0, 0), go.BLACK))

        # make it a true eye by turning the corner (1, 1) into an eye itself
        gs.do_move((1, 2), go.BLACK)
        gs.do_move((2, 1), go.BLACK)
        gs.do_move((2, 2), go.BLACK)
        gs.do_move((0, 2), go.BLACK)

        self.assertTrue(gs.is_eyeish((0, 0), go.BLACK))
        self.assertTrue(gs.is_eye((0, 0), go.BLACK))
        self.assertTrue(gs.is_eye((1, 1), go.BLACK))

    def test_eye_recursion(self):
        # a checkerboard pattern of black is 'technically' all true eyes
        # mutually supporting each other
        gs = GameState(7)
        for x in range(gs.size):
            for y in range(gs.size):
                if (x + y) % 2 == 1:
                    gs.do_move((x, y), go.BLACK)
        self.assertTrue(gs.is_eye((0, 0), go.BLACK))

    def test_liberties_after_capture(self):
        # creates 3x3 black group in the middle, that is then all captured
        # ...then an assertion is made that the resulting liberties after
        # capture are the same as if the group had never been there
        gs_capture = GameState(7)
        gs_reference = GameState(7)
        # add in 3x3 black stones
        for x in range(2, 5):
            for y in range(2, 5):
                gs_capture.do_move((x, y), go.BLACK)
        # surround the black group with white stones
        # and set the same white stones in gs_reference
        for x in range(2, 5):
            gs_capture.do_move((x, 1), go.WHITE)
            gs_capture.do_move((x, 5), go.WHITE)
            gs_reference.do_move((x, 1), go.WHITE)
            gs_reference.do_move((x, 5), go.WHITE)
        gs_capture.do_move((1, 1), go.WHITE)
        gs_reference.do_move((1, 1), go.WHITE)
        for y in range(2, 5):
            gs_capture.do_move((1, y), go.WHITE)
            gs_capture.do_move((5, y), go.WHITE)
            gs_reference.do_move((1, y), go.WHITE)
            gs_reference.do_move((5, y), go.WHITE)

        # board configuration and liberties of gs_capture and of gs_reference should be identical
        self.assertTrue(np.all(gs_reference.board == gs_capture.board))
        self.assertTrue(np.all(gs_reference.liberty_counts == gs_capture.liberty_counts))


if __name__ == '__main__':
    unittest.main()
