import os
import unittest
import numpy as np
from AlphaGo.go import GameState
from AlphaGo.models.value import CNNValue
from AlphaGo.training.reinforcement_value_trainer import FILE_TEST
from AlphaGo.training.reinforcement_value_trainer import FILE_TRAIN
from AlphaGo.training.reinforcement_value_trainer import FILE_METADATA
from AlphaGo.training.reinforcement_value_trainer import FOLDER_WEIGHT
from AlphaGo.training.reinforcement_value_trainer import FILE_VALIDATE
from AlphaGo.training.reinforcement_value_trainer import handle_arguments
from AlphaGo.training.reinforcement_value_trainer import load_indices_from_file
from AlphaGo.training.reinforcement_value_trainer import create_and_save_shuffle_indices


class TestCNNValue(unittest.TestCase):

    def setUp(self):
        self.value = CNNValue(['board', 'ones', 'turns_since'])

    def test_save_load(self):
        self.value.save_model('.tmp.value.json')
        copy = CNNValue.load_model('.tmp.value.json')

        # test that loaded class is also an instance of CNNValue
        self.assertTrue(isinstance(copy, CNNValue))
        os.remove('.tmp.value.json')

    # test shape
    def test_ouput_shape(self):
        gs = GameState()

        val = self.value.eval_state(gs)
        self.assertTrue(isinstance(val, np.float64))

    def testTrain(self):
        model = 'tests/test_data/minimodel_value.json'
        data = 'tests/test_data/hdf5/value_training_features.hdf5'
        output = 'tests/test_data/.tmp.training/'
        args = ['train', output, model, data, '--epochs', '1', '-l', '160']
        handle_arguments(args)

        # remove temporary files
        os.remove(os.path.join(output, FILE_METADATA))
        os.remove(os.path.join(output, FILE_TRAIN))
        os.remove(os.path.join(output, FILE_VALIDATE))
        os.remove(os.path.join(output, FILE_TEST))
        os.remove(os.path.join(output, FOLDER_WEIGHT, 'weights.00000.hdf5'))
        os.rmdir(os.path.join(output, FOLDER_WEIGHT))
        os.rmdir(output)

    def testResumeLearning(self):
        model = 'tests/test_data/minimodel_value.json'
        data = 'tests/test_data/hdf5/value_training_features.hdf5'
        output = 'tests/test_data/.tmp.resume-training/'
        args = ['train', output, model, data, '--epochs', '1', '-l', '160']
        handle_arguments(args)

        # resume learning
        args = ['train', output, model, data, '--epochs', '2', '-l', '160',
                '--weights', 'weights.00000.hdf5']
        handle_arguments(args)

        # remove temporary files
        os.remove(os.path.join(output, FILE_METADATA))
        os.remove(os.path.join(output, FILE_TRAIN))
        os.remove(os.path.join(output, FILE_VALIDATE))
        os.remove(os.path.join(output, FILE_TEST))
        os.remove(os.path.join(output, FOLDER_WEIGHT, 'weights.00000.hdf5'))
        os.remove(os.path.join(output, FOLDER_WEIGHT, 'weights.00001.hdf5'))
        os.rmdir(os.path.join(output, FOLDER_WEIGHT))
        os.rmdir(output)

    def testStateAmount(self):
        output = 'tests/test_data/.tmp.state-amount/'
        # create folder
        if not os.path.exists(output):
            os.makedirs(output)

        # shuffle file locations for train/validation/test set
        shuffle_file_train = os.path.join(output, FILE_TRAIN)
        shuffle_file_val = os.path.join(output, FILE_VALIDATE)
        shuffle_file_test = os.path.join(output, FILE_TEST)

        # create shuffle files
        create_and_save_shuffle_indices([.9, .05, .05], 1000000000,
                                        1033, shuffle_file_train,
                                        shuffle_file_val, shuffle_file_test)

        # load from .npz files
        # load training set
        train_indices = load_indices_from_file(shuffle_file_train)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in train_indices}))
        self.assertTrue(unique_indices == 7438)

        # load validation set
        val_indices = load_indices_from_file(shuffle_file_val)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in val_indices}))
        self.assertTrue(unique_indices == 413)

        # load training set
        test_indices = load_indices_from_file(shuffle_file_test)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in test_indices}))
        self.assertTrue(unique_indices == 413)

        # combine all set rows
        combined_indices = np.concatenate((train_indices, val_indices, test_indices), axis=0)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in combined_indices}))
        self.assertTrue(unique_indices == (7438 + 413 + 413))

        # remove temporary files
        os.remove(shuffle_file_train)
        os.remove(shuffle_file_val)
        os.remove(shuffle_file_test)
        os.rmdir(output)


if __name__ == '__main__':
    unittest.main()
