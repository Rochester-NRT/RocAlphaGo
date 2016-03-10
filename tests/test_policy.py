from AlphaGo.models.policy import CNNPolicy
from AlphaGo.go import GameState
import unittest

class TestCNNPolicy(unittest.TestCase):

	def test_default_policy(self):
		policy = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"])
		policy.eval_state(GameState())
		# just hope nothing breaks

	def test_output_size(self):
		policy19 = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"], board=19)
		output = policy19.forward([policy19.preprocessor.state_to_tensor(GameState(19))])
		self.assertEqual(output.shape, (19,19))

		policy13 = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"], board=13)
		output = policy13.forward([policy13.preprocessor.state_to_tensor(GameState(13))])
		self.assertEqual(output.shape, (13,13))

if __name__ == '__main__':
	unittest.main()