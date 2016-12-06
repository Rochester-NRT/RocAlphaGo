import os
from AlphaGo.training.reinforcement_policy_trainer import run_training, log_loss, run_n_games
import unittest
import numpy as np
import numpy.testing as npt
import AlphaGo.go as go
from keras.optimizers import SGD
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.util import sgf_iter_states


class MockPlayer(object):

    def __init__(self, policy):
        with open("tests/test_data/sgf/20160312-Lee-Sedol-vs-AlphaGo.sgf", "r") as f:
            sgf_game = f.read()
        self.moves = [move for (_, move, _) in sgf_iter_states(sgf_game)]
        self.policy = policy

    def get_moves(self, states):
        indices = [len(state.history) for state in states]
        return [self.moves[i] if i < len(self.moves) else go.PASS_MOVE for i in indices]


class MockState(go.GameState):

    def __init__(self, predetermined_winner, length, *args, **kwargs):
        super(MockState, self).__init__(*args, **kwargs)
        self.predetermined_winner = predetermined_winner
        self.length = length

    def do_move(self, *args, **kwargs):
        super(MockState, self).do_move(*args, **kwargs)
        if len(self.history) > self.length:
            self.is_end_of_game = True

    def get_winner(self):
        return self.predetermined_winner


class TestReinforcementPolicyTrainer(unittest.TestCase):

    def testTrain(self):
        model = os.path.join('tests', 'test_data', 'minimodel.json')
        init_weights = os.path.join('tests', 'test_data', 'hdf5', 'random_minimodel_weights.hdf5')
        output = os.path.join('tests', 'test_data', '.tmp.rl.training/')
        args = [model, init_weights, output, '--game-batch', '1', '--iterations', '1']
        run_training(args)

        os.remove(os.path.join(output, 'metadata.json'))
        os.remove(os.path.join(output, 'weights.00000.hdf5'))
        os.remove(os.path.join(output, 'weights.00001.hdf5'))
        os.rmdir(output)

    def testGradientDirectionChangesWithGameResult(self):

        def run_and_get_new_weights(init_weights, winners):
            # Create "mock" states that end after 50 moves with a predetermined winner.
            states = [MockState(winner, 50, size=19) for winner in winners]

            policy1 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            policy2 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            policy1.model.set_weights(init_weights)
            optimizer = SGD(lr=0.001)
            policy1.model.compile(loss=log_loss, optimizer=optimizer)

            learner = MockPlayer(policy1)
            opponent = MockPlayer(policy2)

            # Run RL training
            run_n_games(optimizer, learner, opponent, 2, mock_states=states)

            return policy1.model.get_weights()

        policy = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
        initial_parameters = policy.model.get_weights()
        # Cases 1 and 2 have identical starting models and identical (state, action) pairs,
        # but they differ in who won the games.
        parameters1 = run_and_get_new_weights(initial_parameters, [go.BLACK, go.WHITE])
        parameters2 = run_and_get_new_weights(initial_parameters, [go.WHITE, go.BLACK])

        # Assert that some parameters changed.
        any_change_1 = any(not np.array_equal(i, p1) for (i, p1) in zip(initial_parameters,
                                                                        parameters1))
        any_change_2 = any(not np.array_equal(i, p2) for (i, p2) in zip(initial_parameters,
                                                                        parameters2))
        self.assertTrue(any_change_1)
        self.assertTrue(any_change_2)

        # Changes in case 1 should be equal and opposite to changes in case 2. Allowing 0.1%
        # difference in precision.
        for (i, p1, p2) in zip(initial_parameters, parameters1, parameters2):
            diff1 = p1 - i
            diff2 = p2 - i
            npt.assert_allclose(diff1, -diff2, rtol=1e-3, atol=1e-11)

    def testRunNGamesUpdatesWeights(self):
        policy1 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
        policy2 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
        learner = MockPlayer(policy1)
        opponent = MockPlayer(policy2)
        optimizer = SGD()
        init_weights = policy1.model.get_weights()
        policy1.model.compile(loss=log_loss, optimizer=optimizer)

        # Run RL training
        run_n_games(optimizer, learner, opponent, 2)

        # Get new weights for comparison
        trained_weights = policy1.model.get_weights()

        # Assert that some parameters changed.
        any_change = any(not np.array_equal(i, t) for (i, t) in zip(init_weights, trained_weights))
        self.assertTrue(any_change)


if __name__ == '__main__':
    unittest.main()
