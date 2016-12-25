from AlphaGo.models.value import CNNValue
from AlphaGo.go import GameState
import unittest
import os


class TestCNNValue(unittest.TestCase):

	def setUp(self):
		self.value = CNNValue(['board', 'ones', 'turns_since'])

	def test_save_load(self):
		self.value.save_model('.tmp.value.json')
		copy = CNNValue.load_model('.tmp.value.json')

		# test that loaded class is also an instance of CNNValue
		self.assertTrue(isinstance(copy, CNNValue))
		os.remove('.tmp.value.json')

	def test_ouput_shape(self):
		gs = GameState()

		val = self.value.eval_state(gs)
		self.assertEqual(len(val), 1)


if __name__ == '__main__':
	unittest.main()