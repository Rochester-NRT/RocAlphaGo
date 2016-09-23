from AlphaGo.models.policy import CNNPolicy, ResnetPolicy
from AlphaGo import go
from AlphaGo.go import GameState
from AlphaGo.ai import GreedyPolicyPlayer, ProbabilisticPolicyPlayer
import numpy as np
import unittest
import os


class TestCNNPolicy(unittest.TestCase):

    def test_default_policy(self):
        policy = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"])
        policy.eval_state(GameState())
        # just hope nothing breaks

    def test_batch_eval_state(self):
        policy = CNNPolicy(["board", "liberties", "sensibleness", "capture_size"])
        results = policy.batch_eval_state([GameState(), GameState()])
        self.assertEqual(len(results), 2)  # one result per GameState
        self.assertEqual(len(results[0]), 361)  # each one has 361 (move,prob) pairs

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
        policy.model.save_weights(weights_file, overwrite=True)
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


class TestResnetPolicy(unittest.TestCase):
    def test_default_policy(self):
        policy = ResnetPolicy(["board", "liberties", "sensibleness", "capture_size"])
        policy.eval_state(GameState())
        # just hope nothing breaks

    def test_batch_eval_state(self):
        policy = ResnetPolicy(["board", "liberties", "sensibleness", "capture_size"])
        results = policy.batch_eval_state([GameState(), GameState()])
        self.assertEqual(len(results), 2)  # one result per GameState
        self.assertEqual(len(results[0]), 361)  # each one has 361 (move,prob) pairs

    def test_save_load(self):
        """Identical to above test_save_load
        """
        policy = ResnetPolicy(["board", "liberties", "sensibleness", "capture_size"])

        model_file = 'TESTPOLICY.json'
        weights_file = 'TESTWEIGHTS.h5'
        model_file2 = 'TESTPOLICY2.json'
        weights_file2 = 'TESTWEIGHTS2.h5'

        # test saving model/weights separately
        policy.save_model(model_file)
        policy.model.save_weights(weights_file, overwrite=True)
        # test saving them together
        policy.save_model(model_file2, weights_file2)

        copypolicy = ResnetPolicy.load_model(model_file)
        copypolicy.model.load_weights(weights_file)

        copypolicy2 = ResnetPolicy.load_model(model_file2)

        for w1, w2 in zip(copypolicy.model.get_weights(), copypolicy2.model.get_weights()):
            self.assertTrue(np.all(w1 == w2))

        # check that save/load keeps the ResnetPolicy class
        self.assertTrue(type(policy) == type(copypolicy))

        os.remove(model_file)
        os.remove(weights_file)
        os.remove(model_file2)
        os.remove(weights_file2)


class TestPlayers(unittest.TestCase):

    def test_greedy_player(self):
        gs = GameState()
        policy = CNNPolicy(["board", "ones", "turns_since"])
        player = GreedyPolicyPlayer(policy)
        for i in range(20):
            move = player.get_move(gs)
            self.assertIsNotNone(move)
            gs.do_move(move)

    def test_probabilistic_player(self):
        gs = GameState()
        policy = CNNPolicy(["board", "ones", "turns_since"])
        player = ProbabilisticPolicyPlayer(policy)
        for i in range(20):
            move = player.get_move(gs)
            self.assertIsNotNone(move)
            gs.do_move(move)

    def test_sensible_probabilistic(self):
        gs = GameState()
        policy = CNNPolicy(["board", "ones", "turns_since"])
        player = ProbabilisticPolicyPlayer(policy)
        empty = (10, 10)
        for x in range(19):
            for y in range(19):
                if (x, y) != empty:
                    gs.do_move((x, y), go.BLACK)
        gs.current_player = go.BLACK
        self.assertIsNone(player.get_move(gs))

    def test_sensible_greedy(self):
        gs = GameState()
        policy = CNNPolicy(["board", "ones", "turns_since"])
        player = GreedyPolicyPlayer(policy)
        empty = (10, 10)
        for x in range(19):
            for y in range(19):
                if (x, y) != empty:
                    gs.do_move((x, y), go.BLACK)
        gs.current_player = go.BLACK
        self.assertIsNone(player.get_move(gs))


if __name__ == '__main__':
    unittest.main()
