from AlphaGo.models.game_converter import game_converter
import unittest


class TestGameState(unittest.TestCase):
	def setUp(self):
		self.gc = game_converter()

	def test_batch_convert(self):
		sample_generator = self.gc.batch_convert(
			"tests/test_sgfs",
			features=[
				"board", "ones", "turns_since", "liberties", "capture_size",
				"self_atari_size", "liberties_after", "sensibleness", "zeros"])
		for sample in sample_generator:
			self.assertIsNot(sample, None)

if __name__ == '__main__':
	unittest.main()
