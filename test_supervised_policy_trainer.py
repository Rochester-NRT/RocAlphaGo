import os
from AlphaGo.training.supervised_policy_trainer import run_training
import unittest


class TestSupervisedPolicyTrainer(unittest.TestCase):
	def testTrain(self):
		model = 'D:\\ps\\club\\Go\\models\\all_feat_model.json'
		data = 'd:\\tmp\\1step.33.h5'
		output = 'tests/test_data/.tmp.training/'
		args = [model, data, output, '--epochs', '60', '--epoch-length', '80', '--learning-rate', '0.2', '-B', '1']
		run_training(args)

		#os.remove(os.path.join(output, 'metadata.json'))
		#os.remove(os.path.join(output, 'shuffle.npz'))
		#os.remove(os.path.join(output, 'weights.00000.hdf5'))
		#os.rmdir(output)

if __name__ == '__main__':
	unittest.main()
