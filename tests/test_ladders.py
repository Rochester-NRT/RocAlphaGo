from AlphaGo.go import GameState
import AlphaGo.go as go
import numpy as np
import unittest

class TestLadders(unittest.TestCase):

    def setUp(self):
        self.s = GameState()
        self.s.do_move((1, 1))  # Black 1
        self.s.do_move((2, 1))  # White 2
        self.s.do_move((2, 0))  # Black 3
        self.s.do_move((4, 4))  # White 4, creates ladder escape
        self.s.do_move((2, 2))  # Black 5
        self.s.do_move((6, 1))  # White 6
        # self.s.do_move((3, 0))  # Black 7, first ladder capture
        # self.s.do_move((3, 1))  # White 8
        # self.s.do_move((4, 1))  # Black 9
        # self.s.do_move((3, 2))  # White 10
        # self.s.do_move((3, 3))  # Black 11
        # self.s.do_move((4, 2))  # White 12, ladder escape
        # self.s.do_move((5, 2))  # Black 13, not ladder capture because of white (4, 4)
        # self.s.do_move((4, 3))  # White 14

    def basic_capture(self):
        self.setUp()
        self.assertTrue(self.s.is_ladder_capture((3, 0)))
        self.s.do_move((3, 0))
        self.s.do_move((3, 1))
        self.assertTrue(self.s.is_ladder_capture((4, 1)))
        self.s.do_move((4, 1))
        self.s.do_move((3, 2))
        self.assertTrue(self.s.is_ladder_capture((3, 3)))
        self.s.do_move((3, 3))
        self.s.do_move((4, 2))
        self.assertFalse(self.s.is_ladder_capture((5, 2)))

    def basic_escape(self):
        self.setUp()
        self.s.do_move((3, 0))
        self.assertFalse(self.s.is_ladder_escape((3, 1)))
        self.s.do_move((3, 1))
        self.s.do_move((4, 1))
        self.assertFalse(self.s.is_ladder_escape((3, 2)))
        self.s.do_move((3, 2))
        self.s.do_move((3, 3))
        self.assertTrue(self.s.is_ladder_escape((4, 2)))



if __name__ == '__main__':
    unittest.main()
