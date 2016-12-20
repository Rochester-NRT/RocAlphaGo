import os
from AlphaGo.training.supervised_policy_trainer import run_training
from AlphaGo.training.supervised_policy_trainer import load_indices_from_file
import unittest
import numpy as np


class TestSupervisedPolicyTrainer(unittest.TestCase):
    def testTrain(self):
        model = 'tests/test_data/minimodel.json'
        data = 'tests/test_data/hdf5/alphago-vs-lee-sedol-features.hdf5'
        output = 'tests/test_data/.tmp.training/'
        args = [model, data, output, '--epochs', '1', '-l', '160']
        run_training(args)

        # remove temporary files
        os.remove(os.path.join(output, 'metadata.json'))
        os.remove(os.path.join(output, 'shuffle_train.npz'))
        os.remove(os.path.join(output, 'shuffle_validate.npz'))
        os.remove(os.path.join(output, 'shuffle_test.npz'))
        os.remove(os.path.join(output, 'weights.00000.hdf5'))
        os.rmdir(output)

    def testResumeLearning(self):
        model = 'tests/test_data/minimodel.json'
        data = 'tests/test_data/hdf5/alphago-vs-lee-sedol-features.hdf5'
        output = 'tests/test_data/.tmp.resume-training/'
        args = [model, data, output, '--epochs', '1', '-l', '160']
        run_training(args)

        # resume learning
        args = [model, data, output, '--epochs', '1', '-l', '160',
                '--weights', 'weights.00000.hdf5']
        run_training(args)

        # remove temporary files
        os.remove(os.path.join(output, 'metadata.json'))
        os.remove(os.path.join(output, 'shuffle_train.npz'))
        os.remove(os.path.join(output, 'shuffle_validate.npz'))
        os.remove(os.path.join(output, 'shuffle_test.npz'))
        os.remove(os.path.join(output, 'weights.00000.hdf5'))
        os.rmdir(output)

    def testStateAmount(self):
        model = 'tests/test_data/minimodel.json'
        data = 'tests/test_data/hdf5/alphago-vs-lee-sedol-features.hdf5'
        output = 'tests/test_data/.tmp.state-amount/'
        args = [model, data, output, '--epochs', '0']
        run_training(args)

        # shuffle file locations for train/validation/test set
        shuffle_file_train = os.path.join(output, "shuffle_train.npz")
        shuffle_file_val = os.path.join(output, "shuffle_validate.npz")
        shuffle_file_test = os.path.join(output, "shuffle_test.npz")

        # load from .npz files
        # load training set
        train_indices = load_indices_from_file(shuffle_file_train)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in train_indices}))
        self.assertTrue(unique_indices == 7840)

        # load validation set
        val_indices = load_indices_from_file(shuffle_file_val)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in val_indices}))
        self.assertTrue(unique_indices == 413)

        # load training set
        test_indices = load_indices_from_file(shuffle_file_test)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in test_indices}))
        self.assertTrue(unique_indices == 11)

        # combine all set rows
        combined_indices = np.concatenate((train_indices, val_indices, test_indices), axis=0)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in combined_indices}))
        self.assertTrue(unique_indices == (7840 + 413 + 11))

        # remove temporary files
        os.remove(shuffle_file_train)
        os.remove(shuffle_file_val)
        os.remove(shuffle_file_test)
        os.rmdir(output)


if __name__ == '__main__':
    unittest.main()
