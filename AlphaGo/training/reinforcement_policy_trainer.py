import os
import json
import numpy as np
from shutil import copyfile
from keras.optimizers import Optimizer
import keras.backend as K
from AlphaGo.ai import ProbabilisticPolicyPlayer
import AlphaGo.go as go
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.util import flatten_idx


class BatchedReinforcementLearningSGD(Optimizer):
    '''A Keras Optimizer that sums gradients together for each game, applying them only once the
    winner is known.

    It is the responsibility of the calling code to call set_current_game() before each example to
    tell the optimizer for which game gradients should be accumulated, and to call set_result() to
    tell the optimizer what the sign of the gradient for each game should be and when all games are
    over.

    Arguments
        lr: float >= 0. Learning rate.
        ng: int > 0. Number of games played in parallel. Each one has its own cumulative gradient.
    '''

    def __init__(self, lr=0.01, ng=20, **kwargs):
        super(BatchedReinforcementLearningSGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)
        self.cumulative_gradients = []
        self.num_games = ng
        self.game_idx = K.variable(0)  # which gradient to accumulate in the next batch.
        self.gradient_sign = [K.variable(0) for _ in range(ng)]
        self.running_games = K.variable(self.num_games)

    def set_current_game(self, game_idx):
        K.set_value(self.game_idx, game_idx)

    def set_result(self, game_idx, won_game):
        '''Mark the outcome of the game at index game_idx. Once all games are complete, updates
        are automatically triggered in the next call to a keras fit function.
        '''
        K.set_value(self.gradient_sign[game_idx], +1 if won_game else -1)
        # Note: using '-= 1' would create a new variable, which would invalidate the dependencies
        # in get_updates().
        K.set_value(self.running_games, K.get_value(self.running_games) - 1)

    def get_updates(self, params, constraints, loss):
        # Note: get_updates is called *once* by keras. Its job is to return a set of 'update
        # operations' to any K.variable (e.g. model weights or self.num_games). Updates are applied
        # whenever Keras' train_function is evaluated, i.e. in every batch. Model.fit_on_batch()
        # will trigger exactly one update. All updates use the 'old' value of parameters - there is
        # no dependency on the order of the list of updates.
        self.updates = []
        # Get expressions for gradients of model parameters.
        grads = self.get_gradients(loss, params)
        # Create a set of accumulated gradients, one for each game.
        shapes = [K.get_variable_shape(p) for p in params]
        self.cumulative_gradients = [[K.zeros(shape) for shape in shapes]
                                     for _ in range(self.num_games)]

        def conditional_update(cond, variable, new_value):
            '''Helper function to create updates that only happen when cond is True. Writes to
            self.updates and returns the new variable.

            Note: K.update(x, x) is cheap, but K.update_add(x, K.zeros_like(x)) can be expensive.
            '''
            maybe_new_value = K.switch(cond, new_value, variable)
            self.updates.append(K.update(variable, maybe_new_value))
            return maybe_new_value

        # Update cumulative gradient at index game_idx. This is done by returning an update for all
        # gradients that is a no-op everywhere except for the game_idx'th one. When game_idx is
        # changed by a call to set_current_game(), it will change the gradient that is getting
        # accumulated.
        # new_cumulative_gradients keeps references to the updated variables for use below in
        # updating parameters with the freshly-accumulated gradients.
        new_cumulative_gradients = [[None] * len(cgs) for cgs in self.cumulative_gradients]
        for i, cgs in enumerate(self.cumulative_gradients):
            for j, (g, cg) in enumerate(zip(grads, cgs)):
                new_gradient = conditional_update(K.equal(self.game_idx, i), cg, cg + g)
                new_cumulative_gradients[i][j] = new_gradient

        # Compute the net update to parameters, taking into account the sign of each cumulative
        # gradient.
        net_grads = [K.zeros_like(g) for g in grads]
        for i, cgs in enumerate(new_cumulative_gradients):
            for j, cg in enumerate(cgs):
                net_grads[j] += self.gradient_sign[i] * cg

        # Trigger a full update when all games have finished.
        self.trigger_update = K.lesser_equal(self.running_games, 0)

        # Update model parameters conditional on trigger_update.
        for p, g in zip(params, net_grads):
            new_p = p + g * self.lr
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            conditional_update(self.trigger_update, p, new_p)

        # 'reset' game counter and gradient signs when parameters are updated.
        for sign in self.gradient_sign:
            conditional_update(self.trigger_update, sign, K.variable(0))
        conditional_update(self.trigger_update, self.running_games, K.variable(self.num_games))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'ng': self.num_games}
        base_config = super(BatchedReinforcementLearningSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _make_training_pair(st, mv, preprocessor):
    # Convert move to one-hot
    st_tensor = preprocessor.state_to_tensor(st)
    mv_tensor = np.zeros((1, st.size * st.size))
    mv_tensor[(0, flatten_idx(mv, st.size))] = 1
    return (st_tensor, mv_tensor)


def run_n_games(optimizer, learner, opponent, num_games):
    '''Run num_games games to completion, calling train_batch() on each position
    the learner sees.

    (Note: optimizer only accumulates gradients in its update function until
    all games have finished)

    '''
    board_size = learner.policy.model.input_shape[-1]
    states = [GameState(size=board_size) for _ in range(num_games)]
    learner_net = learner.policy.model

    # Start all odd games with moves by 'opponent'. Even games will have 'learner' black.
    learner_color = [go.BLACK if i % 2 == 0 else go.WHITE for i in range(num_games)]
    odd_states = states[1::2]
    moves = opponent.get_moves(odd_states)
    for st, mv in zip(odd_states, moves):
        st.do_move(mv)

    current = learner
    other = opponent
    # Need to keep track of the index of unfinished states so that we can communicate which one is
    # being updated to the optimizer.
    idxs_to_unfinished_states = {i: states[i] for i in range(num_games)}
    while len(idxs_to_unfinished_states) > 0:
        # Get next moves by current player for all unfinished states.
        moves = current.get_moves(idxs_to_unfinished_states.values())
        just_finished = []
        # Do each move to each state in order.
        for (idx, state), mv in zip(idxs_to_unfinished_states.iteritems(), moves):
            # Order is important here. We must first get the training pair on the unmodified state.
            # Next, the state is updated and checked to see if the game is over. If it is over, the
            # optimizer is notified via set_result. Finally, train_on_batch is called, which
            # will trigger an update of all parameters only if set_result() has been called
            # for all games already (so set_result must come before train_on_batch).
            is_learnable = current is learner and mv is not go.PASS_MOVE
            if is_learnable:
                (X, y) = _make_training_pair(state, mv, learner.policy.preprocessor)
            state.do_move(mv)
            if state.is_end_of_game:
                learner_is_winner = state.get_winner() == learner_color[idx]
                optimizer.set_result(idx, learner_is_winner)
                just_finished.append(idx)
            if is_learnable:
                optimizer.set_current_game(idx)
                learner_net.train_on_batch(X, y)

        # Remove games that have finished from dict.
        for idx in just_finished:
            del idxs_to_unfinished_states[idx]

        # Swap 'current' and 'other' for next turn.
        current, other = other, current

    # Return the win ratio.
    wins = sum(state.get_winner() == pc for (state, pc) in zip(states, learner_color))
    return float(wins) / num_games


def run_training(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Perform reinforcement learning to improve given policy network. Second phase of pipeline.')  # noqa: E501
    parser.add_argument("model_json", help="Path to policy model JSON.")
    parser.add_argument("initial_weights", help="Path to HDF5 file with inital weights (i.e. result of supervised training).")  # noqa: E501
    parser.add_argument("out_directory", help="Path to folder where the model params and metadata will be saved after each epoch.")  # noqa: E501
    parser.add_argument("--learning-rate", help="Keras learning rate (Default: 0.001)", type=float, default=0.001)  # noqa: E501
    parser.add_argument("--policy-temp", help="Distribution temperature of players using policies (Default: 0.67)", type=float, default=0.67)  # noqa: E501
    parser.add_argument("--save-every", help="Save policy as a new opponent every n batches (Default: 500)", type=int, default=500)  # noqa: E501
    parser.add_argument("--game-batch", help="Number of games per mini-batch (Default: 20)", type=int, default=20)  # noqa: E501
    parser.add_argument("--move-limit", help="Maximum number of moves per game", type=int, default=500)  # noqa: E501
    parser.add_argument("--iterations", help="Number of training batches/iterations (Default: 10000)", type=int, default=10000)  # noqa: E501
    parser.add_argument("--resume", help="Load latest weights in out_directory and resume", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # Baseline function (TODO) default lambda state: 0  (receives either file
    # paths to JSON and weights or None, in which case it uses default baseline 0)
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    ZEROTH_FILE = "weights.00000.hdf5"

    if args.resume:
        if not os.path.exists(os.path.join(args.out_directory, "metadata.json")):
            raise ValueError("Cannot resume without existing output directory")

    if not os.path.exists(args.out_directory):
        if args.verbose:
            print "creating output directory {}".format(args.out_directory)
        os.makedirs(args.out_directory)

    if not args.resume:
        # make a copy of weights file, "weights.00000.hdf5" in the output directory
        copyfile(args.initial_weights, os.path.join(args.out_directory, ZEROTH_FILE))
        if args.verbose:
            print "copied {} to {}".format(args.initial_weights,
                                           os.path.join(args.out_directory, ZEROTH_FILE))
        player_weights = ZEROTH_FILE
    else:
        # if resuming, we expect initial_weights to be just a
        # "weights.#####.hdf5" file, not a full path
        args.initial_weights = os.path.join(args.out_directory,
                                            os.path.basename(args.initial_weights))
        if not os.path.exists(args.initial_weights):
            raise ValueError("Cannot resume; weights {} do not exist".format(args.initial_weights))
        elif args.verbose:
            print "Resuming with weights {}".format(args.initial_weights)
        player_weights = os.path.basename(args.initial_weights)

    # Set initial conditions
    policy = CNNPolicy.load_model(args.model_json)
    policy.model.load_weights(args.initial_weights)
    player = ProbabilisticPolicyPlayer(policy, temperature=args.policy_temp,
                                       move_limit=args.move_limit)

    # different opponents come from simply changing the weights of 'opponent.policy.model'. That
    # is, only 'opp_policy' needs to be changed, and 'opponent' will change.
    opp_policy = CNNPolicy.load_model(args.model_json)
    opponent = ProbabilisticPolicyPlayer(opp_policy, temperature=args.policy_temp,
                                         move_limit=args.move_limit)

    if args.verbose:
        print "created player and opponent with temperature {}".format(args.policy_temp)

    if not args.resume:
        metadata = {
            "model_file": args.model_json,
            "init_weights": args.initial_weights,
            "learning_rate": args.learning_rate,
            "temperature": args.policy_temp,
            "game_batch": args.game_batch,
            "opponents": [ZEROTH_FILE],  # which weights from which to sample an opponent each batch
            "win_ratio": {}  # map from player to tuple of (opponent, win ratio) Useful for
                             # validating in lieu of 'accuracy/loss'
        }
    else:
        with open(os.path.join(args.out_directory, "metadata.json"), "r") as f:
            metadata = json.load(f)

    # Append args of current run to history of full command args.
    metadata["cmd_line_args"] = metadata.get("cmd_line_args", []).append(vars(args))

    def save_metadata():
        with open(os.path.join(args.out_directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)

    optimizer = BatchedReinforcementLearningSGD(lr=args.learning_rate, ng=args.game_batch)
    player.policy.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    for i_iter in xrange(1, args.iterations + 1):
        # Randomly choose opponent from pool (possibly self), and playing
        # game_batch games against them.
        opp_weights = np.random.choice(metadata["opponents"])
        opp_path = os.path.join(args.out_directory, opp_weights)

        # Load new weights into opponent's network, but keep the same opponent object.
        opponent.policy.model.load_weights(opp_path)
        if args.verbose:
            print "Batch {}\tsampled opponent is {}".format(i_iter, opp_weights)

        # Run games (and learn from results). Keep track of the win ratio vs
        # each opponent over time.
        win_ratio = run_n_games(optimizer, player, opponent, args.game_batch)
        metadata["win_ratio"][player_weights] = (opp_weights, win_ratio)

        # Save all intermediate models.
        player_weights = "weights.%05d.hdf5" % i_iter
        player.policy.model.save_weights(os.path.join(args.out_directory, player_weights))

        # Add player to batch of oppenents once in a while.
        if i_iter % args.save_every == 0:
            metadata["opponents"].append(player_weights)
        save_metadata()

if __name__ == '__main__':
    run_training()
