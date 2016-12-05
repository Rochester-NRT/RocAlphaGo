import os
from AlphaGo.training.reinforcement_policy_trainer import \
    run_training, _make_training_pair, BatchedReinforcementLearningSGD, log_loss, run_n_games
import unittest
import numpy as np
import numpy.testing as npt
import keras.backend as K
import AlphaGo.go as go
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


class TestOptimizer(unittest.TestCase):

    def testApplyAndResetOnGamesFinished(self):
        policy = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
        state = go.GameState(size=19)
        optimizer = BatchedReinforcementLearningSGD(lr=0.01, ng=2)
        policy.model.compile(loss=log_loss, optimizer=optimizer)

        # Helper to check initial conditions of the optimizer.
        def assertOptimizerInitialConditions():
            for v in optimizer.gradient_sign:
                self.assertEqual(K.eval(v), 0)
            self.assertEqual(K.eval(optimizer.running_games), 2)

        initial_parameters = policy.model.get_weights()

        def assertModelEffect(changed):
            any_change = False
            for cur, init in zip(policy.model.get_weights(), initial_parameters):
                if not np.allclose(init, cur):
                    any_change = True
                    break
            self.assertEqual(any_change, changed)

        assertOptimizerInitialConditions()

        # Make moves on the state and get trainable (state, action) pairs from them.
        state_tensors = []
        action_tensors = []
        moves = [(2, 2), (16, 16), (3, 17), (16, 2), (4, 10), (10, 3)]
        for m in moves:
            (st_tensor, mv_tensor) = _make_training_pair(state, m, policy.preprocessor)
            state_tensors.append(st_tensor)
            action_tensors.append(mv_tensor)
            state.do_move(m)

        for i, (s, a) in enumerate(zip(state_tensors, action_tensors)):
            # Even moves in game 0, odd moves in game 1
            game_idx = i % 2
            optimizer.set_current_game(game_idx)
            is_last_move = i + 2 >= len(moves)
            if is_last_move:
                # Mark game 0 as a win and game 1 as a loss.
                optimizer.set_result(game_idx, game_idx == 0)
            else:
                # Games not finished yet; assert no change to optimizer state.
                assertOptimizerInitialConditions()
            # train_on_batch accumulates gradients, and should only cause a change to parameters
            # on the first call after the final set_result() call
            policy.model.train_on_batch(s, a)
            if i + 1 < len(moves):
                assertModelEffect(changed=False)
            else:
                assertModelEffect(changed=True)
        # Once both games finished, the last call to train_on_batch() should have triggered a reset
        # to the optimizer parameters back to initial conditions.
        assertOptimizerInitialConditions()

    def testGradientDirectionChangesWithGameResult(self):

        def run_and_get_new_weights(init_weights, win0, win1):
            state = go.GameState(size=19)
            policy = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
            policy.model.set_weights(init_weights)
            optimizer = BatchedReinforcementLearningSGD(lr=0.01, ng=2)
            policy.model.compile(loss=log_loss, optimizer=optimizer)

            # Make moves on the state and get trainable (state, action) pairs from them.
            moves = [(2, 2), (16, 16), (3, 17), (16, 2), (4, 10), (10, 3)]
            state_tensors = []
            action_tensors = []
            for m in moves:
                (st_tensor, mv_tensor) = _make_training_pair(state, m, policy.preprocessor)
                state_tensors.append(st_tensor)
                action_tensors.append(mv_tensor)
                state.do_move(m)

            for i, (s, a) in enumerate(zip(state_tensors, action_tensors)):
                # Put even state/action pairs in game 0, odd ones in game 1.
                game_idx = i % 2
                optimizer.set_current_game(game_idx)
                is_last_move = i + 2 >= len(moves)
                if is_last_move:
                    if game_idx == 0:
                        optimizer.set_result(game_idx, win0)
                    else:
                        optimizer.set_result(game_idx, win1)
                # train_on_batch accumulates gradients, and should only cause a change to parameters
                # on the first call after the final set_result() call
                policy.model.train_on_batch(s, a)
            return policy.model.get_weights()

        policy = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
        initial_parameters = policy.model.get_weights()
        # Cases 1 and 2 have identical starting models and identical (state, action) pairs,
        # but they differ in who won the games.
        parameters1 = run_and_get_new_weights(initial_parameters, True, False)
        parameters2 = run_and_get_new_weights(initial_parameters, False, True)

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
            npt.assert_allclose(diff1, -diff2, rtol=1e-3)

    def testRunNGamesUpdatesWeights(self):
        policy1 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
        policy2 = CNNPolicy.load_model(os.path.join('tests', 'test_data', 'minimodel.json'))
        learner = MockPlayer(policy1)
        opponent = MockPlayer(policy2)
        init_weights = policy1.model.get_weights()
        optimizer = BatchedReinforcementLearningSGD(lr=0.01, ng=2)
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
