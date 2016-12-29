from AlphaGo.ai import ProbabilisticPolicyPlayer
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess
import h5py
import numpy as np
import os
import warnings

# default settings
DEFAULT_N_TRAINING_PAIRS = 30000000
DEFAULT_MAX_GAME_DEPTH = 500
DEFAULT_TEMPERATURE_SL = 0.67
DEFAULT_TEMPERATURE_RL = 0.1
DEFAULT_BATCH_SIZE = 2
DEFAULT_FILE_NAME = "value_planes.hdf5"

def init_hdf5(h5f, n_features, bd_size):
    try:
        states = h5f.require_dataset(
            'states',
            dtype=np.uint8,
            shape=(1, n_features, bd_size, bd_size),
            maxshape=(None, n_features, bd_size, bd_size),
            # 'None' dimension allows it to grow arbitrarily
            exact=False,
            # allow non-uint8 datasets to be loaded, coerced to uint8
            chunks=(1024, n_features, bd_size, bd_size),
            # approximately 10MB chunks (bigger for more compression,
            # OK because accessed *in order*)
            compression="lzf")
        winners = h5f.require_dataset(
            'winners',
            dtype=np.int8,
            shape=(1, 1),
            maxshape=(None, 1),
            exact=False,
            chunks=(1024, 1),
            compression="lzf")
    except Exception as e:
        os.remove(tmp_file)
        raise e
    return states, winners


def play_batch(player_RL, player_SL, batch_size, features):
    """Play a batch of games in parallel and return one training pair from each game.

    As described in Silver et al, the method for generating value net training data is as follows:

    * pick a number between 1 and 450
    * use the supervised-learning policy to play a game against itself up to that number of moves.
    * now go off-policy and pick a totally random move
    * play out the rest of the game with the reinforcement-learning policy
    * save the state that occurred *right after* the random move,
    * and the end result of the game, as the training pair
    """

    def do_move(states, moves):
        for st, mv in zip(states, moves):
            if not st.is_end_of_game:
                # Only do more moves if not end of game already
                st.do_move(mv)
        return states

    def do_rand_move(states):
        """Do a uniform-random move over legal moves and record info for
        training. Only gets called once per game.
        """

        # Record player color
        colors = [st.current_player for st in states]
        
        # get legal moves and play one at random
        legal_moves = [st.get_legal_moves() for st in states]
        rand_moves = [lm[np.random.choice(len(lm))] for lm in legal_moves]
        states = do_move(states, rand_moves)
        
        # copy all states, these are the generated training data
        training_state_list = [st.copy() for st in states]  # For later 1hot preprocessing
        return training_state_list, colors, states

    def convert(state_list, preprocessor):
        """Convert states to 1-hot and concatenate. X's are game state objects.
        """
        states = np.concatenate(
            [preprocessor.state_to_tensor(state) for state in state_list], axis=0)
        return states

    # Lists of game training pairs (1-hot)
    preprocessor = Preprocess(features)
    player = player_SL
    states = [GameState() for _ in xrange(batch_size)]

    # Randomly choose turn to play uniform random. Move prior will be from SL
    # policy. Moves after will be from RL policy.    
    i_rand_move = np.random.choice(range(449))
    print("Random move: " + str(i_rand_move)) 

    for _ in xrange(i_rand_move):
        # Get moves (batch)
        batch_moves = player.get_moves(states)
        # Do moves (black)
        states = do_move(states, batch_moves)

    # Make random move
    states_list, colors, states = do_rand_move(states)

    # switch to player_RL policy
    player = player_RL
    
    # TODO check that random move is played before game is finished?
    # remove state if so?

    while True:
        # Get moves (batch)
        batch_moves = player.get_moves(states)
        # Do moves (black)
        states = do_move(states, batch_moves)

        # check if all games are finished
        done = [st.is_end_of_game for st in states]
        
        if all(done):
            break

    # Concatenate training examples
    training_states = convert(states_list, preprocessor)
    # get winners list
    winners = np.array([st.get_winner() for st in states]).reshape(batch_size, 1)
    return training_states, winners


def generate_data(player_RL, player_SL, hdf5_file, n_training_pairs,
                  batch_size, bd_size, features, verbose):
    # used features
    n_features = Preprocess(features).output_dim
    # temporary hdf5 file
    tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
    # open hdf5 file
    h5f = h5py.File(tmp_file, 'w')
    # initialize a new hdf5 file
    h5_states, h5_winners = init_hdf5(h5f, n_features, bd_size)
    
    if verbose:
        print(str(hdf5_file) + " file initialized.")

    next_idx = 0
    while True:
        states, winners = play_batch(player_RL, player_SL, batch_size, features)
        if states is not None:
            try:
                # get actual batch size in case any pair was removed
                actual_batch_size = len(states)
                
                # add states and winners to hdf5 file
                h5_states.resize((next_idx + actual_batch_size, n_features, bd_size, bd_size))
                h5_winners.resize((next_idx + actual_batch_size, 1))
                h5_states[next_idx:] = states
                h5_winners[next_idx:] = winners
                
                # count saved pairs
                next_idx += actual_batch_size
            except Exception as e:
                warnings.warn("Unknown error occured during batch save to HDF5 file: {}".format(hdf5_file))  # noqa: E501
                raise e

        if verbose and next_idx % (batch_size * 10000):
            print("Generated " + str(next_idx) + " training pairs")

        # stop data generation when at least n_trainings_pairs have been created
        if n_training_pairs >= next_idx:
            break

    # processing complete: rename tmp_file to hdf5_file
    h5f.close()
    os.rename(tmp_file, hdf5_file)
    if verbose:
        print("Value training data succesfull created.")


def handle_arguments(cmd_line_args=None):
    """Run generate data. command-line args may be passed in as a list
    """

    import argparse
    parser = argparse.ArgumentParser(description='Play games used for training'
                                     'value network (third phase of pipeline). '
                                     'The final policy from the RL phase plays '
                                     'against itself and training pairs for value '
                                     'network are generated from the outcome in each '
                                     'games, following an off-policy, uniform random move')
    # required arguments
    parser.add_argument("SL_weights_path", help="Path to file with supervised "
                        "learning policy weights.")
    parser.add_argument("RL_weights_path", help="Path to file with reinforcement "
                        "learning policy weights.")
    parser.add_argument("model_path", help="Path to network architecture file.")
    # optional arguments
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file) Default: " + DEFAULT_FILE_NAME, default=DEFAULT_FILE_NAME)  # noqa: E501
    parser.add_argument("--n_training_pairs", help="Number of training pairs to generate. Default: " + str(DEFAULT_N_TRAINING_PAIRS), type=int, default=DEFAULT_N_TRAINING_PAIRS)  # noqa: E501
    parser.add_argument("--batch_size", help="Number of games to run in parallel. Default: " + str(DEFAULT_BATCH_SIZE), type=int, default=DEFAULT_BATCH_SIZE)  # noqa: E501
    parser.add_argument("--features", "-f", help="Comma-separated list of features to compute and store or 'all'. Default: all", default='all')  # noqa: E501
    parser.add_argument("--sl-temperature", help="Distribution temperature of players using SL policies. Default: " + str(DEFAULT_TEMPERATURE_SL), type=float, default=DEFAULT_TEMPERATURE_SL)  # noqa: E501
    parser.add_argument("--rl-temperature", help="Distribution temperature of players using RL policies. Default: " + str(DEFAULT_TEMPERATURE_RL), type=float, default=DEFAULT_TEMPERATURE_RL)  # noqa: E501

    # show help or parse arguments
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # list with features used for value network
    # features = policy_SL.preprocessor.feature_list
    if args.features.lower() == 'all':
        features = [
            "board",
            "ones",
            "turns_since",
            "liberties",
            "capture_size",
            "self_atari_size",
            "liberties_after",
            "ladder_capture",
            "ladder_escape",
            "sensibleness",
            "zeros"]
    else:
        features = args.features.split(",")

    # always add colour feature
    if "color" not in features:
        features.append("color")

    # Load SL architecture and weights from file
    policy_SL = CNNPolicy.load_model(args.model_path)
    policy_SL.model.load_weights(args.SL_weights_path)
    # create SL player
    player_SL = ProbabilisticPolicyPlayer(policy_SL, temperature=args.sl_temperature, 
                                          move_limit=DEFAULT_MAX_GAME_DEPTH)

    # Load RL architecture and weights from file
    policy_RL = CNNPolicy.load_model(args.model_path)
    policy_RL.model.load_weights(args.RL_weights_path)
    # Create RL player
    player_RL = ProbabilisticPolicyPlayer(policy_RL, temperature=args.rl_temperature, 
                                          move_limit=DEFAULT_MAX_GAME_DEPTH)

    # TODO load board size from model
    # generate data
    generate_data(player_RL, player_SL, args.outfile, args.n_training_pairs,
                  args.batch_size, 19, features, args.verbose)


if __name__ == '__main__':
    handle_arguments()
