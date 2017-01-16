from AlphaGo.models.value import CNNValue
from AlphaGo import go
from AlphaGo.go import GameState
from AlphaGo.ai import GreedyValuePlayer, ProbabilisticValuePlayer
import numpy as np
import unittest
import os


class TestCNNValue(unittest.TestCase):

    def test_default_value(self):
        value = CNNValue(["board", "liberties", "sensibleness", "capture_size"])
        value.eval_state(GameState())
        # just hope nothing breaks

    def test_output_size(self):
        value19 = CNNValue(["board", "liberties", "sensibleness", "capture_size"], board=19)
        output = value19.forward(value19.preprocessor.state_to_tensor(GameState(19)))
        self.assertEqual(output.shape, (1, 1))

        value13 = CNNValue(["board", "liberties", "sensibleness", "capture_size"], board=13)
        output = value13.forward(value13.preprocessor.state_to_tensor(GameState(13)))
        self.assertEqual(output.shape, (1, 1))

    def test_save_load(self):
        value = CNNValue(["board", "liberties", "sensibleness", "capture_size"])

        model_file = 'TESTVALUE.json'
        weights_file = 'TESTWEIGHTS.h5'
        model_file2 = 'TESTVALUE2.json'
        weights_file2 = 'TESTWEIGHTS2.h5'

        # test saving model/weights separately
        value.save_model(model_file)
        value.model.save_weights(weights_file, overwrite=True)
        # test saving them together
        value.save_model(model_file2, weights_file2)

        copyvalue = CNNValue.load_model(model_file)
        copyvalue.model.load_weights(weights_file)

        copyvalue2 = CNNValue.load_model(model_file2)

        for w1, w2 in zip(copyvalue.model.get_weights(), copyvalue2.model.get_weights()):
            self.assertTrue(np.all(w1 == w2))

        os.remove(model_file)
        os.remove(weights_file)
        os.remove(model_file2)
        os.remove(weights_file2)


class TestValuePlayers(unittest.TestCase):

    def test_greedy_player(self):
        gs = GameState(size=9)
        value = CNNValue(["board", "ones", "turns_since"], board=9)
        player = GreedyValuePlayer(value)
        for i in range(10):
            move = player.get_move(gs)
            self.assertIsNotNone(move)
            gs.do_move(move)

    def test_probabilistic_player(self):
        gs = GameState(size=9)
        value = CNNValue(["board", "ones", "turns_since"], board=9)
        player = ProbabilisticValuePlayer(value)
        for i in range(10):
            move = player.get_move(gs)
            self.assertIsNotNone(move)
            gs.do_move(move)

    def test_sensible_probabilistic(self):
        gs = GameState()
        value = CNNValue(["board", "ones", "turns_since"])
        player = ProbabilisticValuePlayer(value)
        empty = (10, 10)
        for x in range(19):
            for y in range(19):
                if (x, y) != empty:
                    gs.do_move((x, y), go.BLACK)
        gs.current_player = go.BLACK
        self.assertIsNone(player.get_move(gs))

    def test_sensible_greedy(self):
        gs = GameState()
        value = CNNValue(["board", "ones", "turns_since"])
        player = GreedyValuePlayer(value)
        empty = (10, 10)
        for x in range(19):
            for y in range(19):
                if (x, y) != empty:
                    gs.do_move((x, y), go.BLACK)
        gs.current_player = go.BLACK
        self.assertIsNone(player.get_move(gs))


if __name__ == '__main__':
    unittest.main()
