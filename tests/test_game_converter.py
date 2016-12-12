from AlphaGo.preprocessing.game_converter import run_game_converter
from AlphaGo.util import sgf_to_gamestate
import unittest
import os


class TestSGFLoading(unittest.TestCase):
    def test_ab_aw(self):
        with open('tests/test_data/sgf_with_handicap/ab_aw.sgf', 'r') as f:
            sgf_to_gamestate(f.read())


class TestCmdlineConverter(unittest.TestCase):

    def test_directory_conversion(self):
        args = ['--features', 'board,ones,turns_since',
                '--outfile', '.tmp.testing.h5',
                '--directory', 'tests/test_data/sgf/']
        run_game_converter(args)
        os.remove('.tmp.testing.h5')

    def test_directory_walk(self):
        args = ['--features', 'board,ones,turns_since',
                '--outfile', '.tmp.testing.h5',
                '--directory', 'tests/test_data', '--recurse']
        run_game_converter(args)
        os.remove('.tmp.testing.h5')


if __name__ == '__main__':
    unittest.main()
