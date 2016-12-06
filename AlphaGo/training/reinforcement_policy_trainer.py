import os
import json
import numpy as np
from shutil import copyfile
from keras.optimizers import SGD
import keras.backend as K
from AlphaGo.ai import ProbabilisticPolicyPlayer
import AlphaGo.go as go
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.util import flatten_idx


def _make_training_pair(st, mv, preprocessor):
    # Convert move to one-hot
    st_tensor = preprocessor.state_to_tensor(st)
    mv_tensor = np.zeros((1, st.size * st.size))
    mv_tensor[(0, flatten_idx(mv, st.size))] = 1
    return (st_tensor, mv_tensor)


def run_n_games(optimizer, learner, opponent, num_games, mock_states=[]):
    '''Run num_games games to completion, keeping track of each position and move of the learner.

    (Note: learning cannot happen until all games have completed)
    '''

    board_size = learner.policy.model.input_shape[-1]
    states = [go.GameState(size=board_size) for _ in range(num_games)]
    learner_net = learner.policy.model

    # Allowing injection of a mock state object for testing purposes
    if mock_states:
        states = mock_states

    # Create one list of features (aka state tensors) and one of moves for each game being played.
    state_tensors = [[] for _ in range(num_games)]
    move_tensors = [[] for _ in range(num_games)]

    # List of booleans indicating whether the 'learner' player won.
    learner_won = [None] * num_games

    # Start all odd games with moves by 'opponent'. Even games will have 'learner' black.
    learner_color = [go.BLACK if i % 2 == 0 else go.WHITE for i in range(num_games)]
    odd_states = states[1::2]
    moves = opponent.get_moves(odd_states)
    for st, mv in zip(odd_states, moves):
        st.do_move(mv)

    current = learner
    other = opponent
    idxs_to_unfinished_states = {i: states[i] for i in range(num_games)}
    while len(idxs_to_unfinished_states) > 0:
        # Get next moves by current player for all unfinished states.
        moves = current.get_moves(idxs_to_unfinished_states.values())
        just_finished = []
        # Do each move to each state in order.
        for (idx, state), mv in zip(idxs_to_unfinished_states.iteritems(), moves):
            # Order is important here. We must get the training pair on the unmodified state before
            # updating it with do_move.
            is_learnable = current is learner and mv is not go.PASS_MOVE
            if is_learnable:
                (st_tensor, mv_tensor) = _make_training_pair(state, mv, learner.policy.preprocessor)
                state_tensors[idx].append(st_tensor)
                move_tensors[idx].append(mv_tensor)
            state.do_move(mv)
            if state.is_end_of_game:
                learner_won[idx] = state.get_winner() == learner_color[idx]
                just_finished.append(idx)

        # Remove games that have finished from dict.
        for idx in just_finished:
            del idxs_to_unfinished_states[idx]

        # Swap 'current' and 'other' for next turn.
        current, other = other, current

    # Train on each game's results, setting the learning rate negative to 'unlearn' positions from
    # games where the learner lost.
    base_lr = optimizer.lr.get_value()
    for (st_tensor, mv_tensor, won) in zip(state_tensors, move_tensors, learner_won):
        optimizer.lr.set_value(base_lr * (+1 if won else -1))
        learner_net.train_on_batch(np.concatenate(st_tensor, axis=0),
                                   np.concatenate(mv_tensor, axis=0))

    # Return the win ratio.
    wins = sum(state.get_winner() == pc for (state, pc) in zip(states, learner_color))
    return float(wins) / num_games


def log_loss(y_true, y_pred):
    '''Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
    taken, and updates with the positive gradient will make that action more likely.
    '''
    return y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()))


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
            print("creating output directory {}".format(args.out_directory))
        os.makedirs(args.out_directory)

    if not args.resume:
        # make a copy of weights file, "weights.00000.hdf5" in the output directory
        copyfile(args.initial_weights, os.path.join(args.out_directory, ZEROTH_FILE))
        if args.verbose:
            print("copied {} to {}".format(args.initial_weights,
                                           os.path.join(args.out_directory, ZEROTH_FILE)))
        player_weights = ZEROTH_FILE
    else:
        # if resuming, we expect initial_weights to be just a
        # "weights.#####.hdf5" file, not a full path
        args.initial_weights = os.path.join(args.out_directory,
                                            os.path.basename(args.initial_weights))
        if not os.path.exists(args.initial_weights):
            raise ValueError("Cannot resume; weights {} do not exist".format(args.initial_weights))
        elif args.verbose:
            print("Resuming with weights {}".format(args.initial_weights))
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
        print("created player and opponent with temperature {}".format(args.policy_temp))

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

    optimizer = SGD(lr=args.learning_rate)
    player.policy.model.compile(loss=log_loss, optimizer=optimizer)
    for i_iter in range(1, args.iterations + 1):
        # Randomly choose opponent from pool (possibly self), and playing
        # game_batch games against them.
        opp_weights = np.random.choice(metadata["opponents"])
        opp_path = os.path.join(args.out_directory, opp_weights)

        # Load new weights into opponent's network, but keep the same opponent object.
        opponent.policy.model.load_weights(opp_path)
        if args.verbose:
            print("Batch {}\tsampled opponent is {}".format(i_iter, opp_weights))

        # Run games (and learn from results). Keep track of the win ratio vs each opponent over
        # time.
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
