from AlphaGo.models.policy import CNNPolicy
from AlphaGo.go import GameState
import numpy as np
import unittest
import os


class TestCNNPolicy(unittest.TestCase):

	def test_default_policy(self):
		policy = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"])
		policy.eval_state(GameState())
		# just hope nothing breaks

	def test_output_size(self):
		policy19 = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"], board=19)
		output = policy19.forward(policy19.preprocessor.state_to_tensor(GameState(19)))
		self.assertEqual(output.shape, (1, 19 * 19))

		policy13 = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"], board=13)
		output = policy13.forward(policy13.preprocessor.state_to_tensor(GameState(13)))
		self.assertEqual(output.shape, (1, 13 * 13))

	def test_save_load(self):
		policy = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"])

		model_file = 'TESTPOLICY.json'
		weights_file = 'TESTWEIGHTS.h5'
		model_file2 = 'TESTPOLICY2.json'
		weights_file2 = 'TESTWEIGHTS2.h5'

		# test saving model/weights separately
		policy.save_model(model_file)
		policy.model.save_weights(weights_file)
		# test saving them together
		policy.save_model(model_file2, weights_file2)

		copypolicy = CNNPolicy.load_model(model_file)
		copypolicy.model.load_weights(weights_file)

		copypolicy2 = CNNPolicy.load_model(model_file2)

		for w1, w2 in zip(copypolicy.model.get_weights(), copypolicy2.model.get_weights()):
			self.assertTrue(np.all(w1 == w2))

		os.remove(model_file)
		os.remove(weights_file)
		os.remove(model_file2)
		os.remove(weights_file2)

if __name__ == '__main__':
	unittest.main()
