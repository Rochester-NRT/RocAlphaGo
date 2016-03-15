from AlphaGo.models.policy import CNNPolicy
from AlphaGo.go import GameState
import unittest
import os


class TestCNNPolicy(unittest.TestCase):

	def test_default_policy(self):
		policy = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"])
		policy.eval_state(GameState())
		# just hope nothing breaks

	def test_output_size(self):
		policy19 = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"], board=19)
		output = policy19.forward([policy19.preprocessor.state_to_tensor(GameState(19))])
		self.assertEqual(output.shape, (19, 19))

		policy13 = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"], board=13)
		output = policy13.forward([policy13.preprocessor.state_to_tensor(GameState(13))])
		self.assertEqual(output.shape, (13, 13))

	def test_save_load(self):
		policy = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"])

		model_file = 'TESTPOLICY.json'
		weights_file = 'TESTWEIGHTS.h5'

		policy.save_model(model_file)
		policy.model.save_weights(weights_file)

		copypolicy = CNNPolicy.load_model(model_file)
		copypolicy.model.load_weights(weights_file)

		os.remove(model_file)
		os.remove(weights_file)

if __name__ == '__main__':
	unittest.main()
