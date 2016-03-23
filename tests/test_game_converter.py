from AlphaGo.preprocessing.game_converter import game_converter
from AlphaGo.util import sgf_to_gamestate
import unittest

class TestSGFLoading(unittest.TestCase):
	def test_ab_aw(self):
		with open('tests/test_data/sgf/ab_aw.sgf', 'r') as f:
			gs = sgf_to_gamestate(f.read())

class TestGameState(unittest.TestCase):
	def setUp(self):
		self.gc = game_converter()

	def test_batch_convert(self):
		sample_generator = self.gc.batch_convert("tests/test_data/sgf",
		features=["board", "ones", "turns_since", "liberties", "capture_size",
		"self_atari_size", "liberties_after", "sensibleness", "zeros"])
		for sample in sample_generator:
			self.assertIsNot(sample, None)

if __name__ == '__main__':
	unittest.main()
