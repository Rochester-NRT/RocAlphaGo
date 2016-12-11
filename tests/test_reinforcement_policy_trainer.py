import os
from AlphaGo.training.reinforcement_policy_trainer import run_training, log_loss, run_n_games
import unittest
import numpy as np
import numpy.testing as npt
import AlphaGo.go as go
from keras.optimizers import SGD
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.util import sgf_iter_states

SGF_FOLDER = "tests/test_data/sgf/"


def _is_sgf(fname):
    return fname.strip()[-4:] == ".sgf"


def _list_mock_games(path):
    """helper function to get all SGF files in a directory (does not recurse)
    """
    files = os.listdir(path)
    return (os.path.join(path, f) for f in files if _is_sgf(f))


def get_sgf_move_probs(sgf_game, policy, player):
    with open(sgf_game, "r") as f:
        sgf_game = f.read()

    def get_single_prob(move, move_probs):
        for (mv, prob) in move_probs:
            if move == mv:
                return prob
        return 0

    return [(move, get_single_prob(move, policy.eval_state(state)))
            for (state, move, pl) in sgf_iter_states(sgf_game) if pl == player]


class MockPlayer(object):

    def __init__(self, policy, sgf_game):
        with open(sgf_game, "r") as f:
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

        def run_and_get_new_weights(init_weights, winners, game):
            # Create "mock" states that end after 2 moves with a predetermined winner.
            states = [MockState(winner, 2, size=19) for winner in winners]

            policy1 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            policy2 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            policy1.model.set_weights(init_weights)
            optimizer = SGD(lr=0.001)
            policy1.model.compile(loss=log_loss, optimizer=optimizer)

            learner = MockPlayer(policy1, game)
            opponent = MockPlayer(policy2, game)

            # Run RL training
            run_n_games(optimizer, learner, opponent, 2, mock_states=states)

            return policy1.model.get_weights()

        def test_game_gradient(game):
            policy = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            initial_parameters = policy.model.get_weights()
            # Cases 1 and 2 have identical starting models and identical (state, action) pairs,
            # but they differ in who won the games.
            parameters1 = run_and_get_new_weights(initial_parameters, [go.BLACK, go.WHITE], game)
            parameters2 = run_and_get_new_weights(initial_parameters, [go.WHITE, go.BLACK], game)

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

        for f in _list_mock_games(SGF_FOLDER):
            test_game_gradient(f)

    def testRunNGamesUpdatesWeights(self):
        def test_game_run_N(game):
            policy1 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            policy2 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            learner = MockPlayer(policy1, game)
            opponent = MockPlayer(policy2, game)
            optimizer = SGD()
            init_weights = policy1.model.get_weights()
            policy1.model.compile(loss=log_loss, optimizer=optimizer)

            # Run RL training
            run_n_games(optimizer, learner, opponent, 2)

            # Get new weights for comparison
            trained_weights = policy1.model.get_weights()

            # Assert that some parameters changed.
            any_change = any(not np.array_equal(i, t)
                             for (i, t) in zip(init_weights, trained_weights))
            self.assertTrue(any_change)

        for f in _list_mock_games(SGF_FOLDER):
            test_game_run_N(f)

    def testWinIncreasesMoveProbability(self):
        def test_game_increase(game):
            # Create "mock" state that ends after 20 moves with the learner winnning
            win_state = [MockState(go.BLACK, 20, size=19)]
            policy1 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            policy2 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            learner = MockPlayer(policy1, game)
            opponent = MockPlayer(policy2, game)
            optimizer = SGD()
            policy1.model.compile(loss=log_loss, optimizer=optimizer)

            # Get initial (before learning) move probabilities for all moves made by black
            init_move_probs = get_sgf_move_probs(game, policy1, go.BLACK)
            init_probs = [prob for (mv, prob) in init_move_probs]

            # Run RL training
            run_n_games(optimizer, learner, opponent, 1, mock_states=win_state)

            # Get new move probabilities for black's moves having finished 1 round of training
            new_move_probs = get_sgf_move_probs(game, policy1, go.BLACK)
            new_probs = [prob for (mv, prob) in new_move_probs]

            # Assert that, on average, move probabilities for black increased having won.
            self.assertTrue(sum((new_probs[i] - init_probs[i]) for i in range(10)) > 0)

        for f in _list_mock_games(SGF_FOLDER):
            test_game_increase(f)

    def testLoseDecreasesMoveProbability(self):
        def test_game_decrease(game):
            # Create "mock" state that ends after 20 moves with the learner losing
            lose_state = [MockState(go.WHITE, 20, size=19)]
            policy1 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            policy2 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            learner = MockPlayer(policy1, game)
            opponent = MockPlayer(policy2, game)
            optimizer = SGD()
            policy1.model.compile(loss=log_loss, optimizer=optimizer)

            # Get initial (before learning) move probabilities for all moves made by black
            init_move_probs = get_sgf_move_probs(game, policy1, go.BLACK)
            init_probs = [prob for (mv, prob) in init_move_probs]

            # Run RL training
            run_n_games(optimizer, learner, opponent, 1, mock_states=lose_state)

            # Get new move probabilities for black's moves having finished 1 round of training
            new_move_probs = get_sgf_move_probs(game, policy1, go.BLACK)
            new_probs = [prob for (mv, prob) in new_move_probs]

            # Assert that, on average, move probabilities for black decreased having lost.
            self.assertTrue(sum((new_probs[i] - init_probs[i]) for i in range(10)) < 0)

        for f in _list_mock_games(SGF_FOLDER):
            test_game_decrease(f)


if __name__ == '__main__':
    unittest.main()
