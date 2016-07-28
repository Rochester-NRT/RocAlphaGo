import os
import argparse
import json
import numpy as np
from shutil import copyfile
from keras.optimizers import SGD
from AlphaGo.ai import ProbabilisticPolicyPlayer
import AlphaGo.go as go
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import flatten_idx
import itertools


def make_training_pairs(player, opp, features, mini_batch_size, board_size=19):
	"""Make training pairs for batch of matches, utilizing player.get_moves (parallel form of
	player.get_move), which calls `CNNPolicy.batch_eval_state`.

	Args:
	player -- player that we're always updating
	opp -- batch opponent
	feature_list -- game features to be one-hot encoded
	mini_batch_size -- number of games in mini-batch

	Return:
	X_list -- list of 1-hot board states associated with moves.
	y_list -- list of 1-hot moves associated with board states.
	winners -- list of winners associated with each game in batch
	"""

	def record_training_pair(st, mv, X, y):
		# Convert move to one-hot
		bsize_flat = bsize * bsize
		state_1hot = preprocessor.state_to_tensor(st)
		move_1hot = np.zeros(bsize_flat)
		move_1hot[flatten_idx(mv, bsize)] = 1
		X.append(state_1hot)
		y.append(move_1hot)

	# First we want to prep the states so that half of the boards get a move from the 'player'.
	# The other half we want to leave alone so that the opponent makes the first move (being black).
	# Decided to alternate every board because this is how humans would play a match.
	def play_half_of_boards(states, moves, X_list, y_list):
		for st, mv, X, y, should_move in zip(states, moves, X_list, y_list, itertools.cycle([True, False])):
			if should_move:
				record_training_pair(st, mv, X, y)
				st.do_move(mv)

	def do_move(states, moves, X_list, y_list, player_color):
		for st, mv, X, y in zip(states, moves, X_list, y_list):
			# Only do more moves if not end of game already
			if not st.is_end_of_game:
				# Only want to record moves by the 'player', not the opponent
				if st.current_player == player_color and mv is not go.PASS_MOVE:
					record_training_pair(st, mv, X, y)
				st.do_move(mv)
		return states, X_list, y_list

	# Lists of game training pairs (1-hot)
	X_list = [list() for _ in xrange(mini_batch_size)]
	y_list = [list() for _ in xrange(mini_batch_size)]
	preprocessor = Preprocess(features)
	bsize = player.policy.model.input_shape[-1]
	states = [GameState(size=board_size) for i in xrange(mini_batch_size)]
	# Randomly choose who goes first (i.e. color of 'player')
	player_color = np.random.choice([go.BLACK, go.WHITE])
	player1, player2 = (player, opp) if player_color == go.BLACK else (opp, player)
	# We let player1 move first for half of the boards in the minibatch
	# The other half of the boards will be empty... waiting for a 'black' player
	moves_player1 = player1.get_moves(states)
	play_half_of_boards(states, moves_player1, X_list, y_list)
	# Now player2 can move and will act as white for half and black for half
	moves_player2 = player2.get_moves(states)
	states, X_list, y_list = do_move(states, moves_player2, X_list, y_list, player_color)
	# Now the game can continue.. each player acting as white and black split across the boards
	while True:
		# Get moves (batch) for player1
		moves_player1 = player1.get_moves(states)
		states, X_list, y_list = do_move(states, moves_player1, X_list, y_list, player_color)
		# Get moves for player2
		moves_player2 = player2.get_moves(states)
		states, X_list, y_list = do_move(states, moves_player2, X_list, y_list, player_color)
		# If all games have ended, we're done. Get winners.
		done = [st.is_end_of_game for st in states]
		if all(done):
			break
	won_game_list = []
	# If player was black, every even board is black, odd board white
	if player_color == go.BLACK:
		for st, game_color in states, itertools.cycle([go.BLACK, go.WHITE]):
			won_game_list.append(st.get_winner() == game_color)
	else:
		for st, game_color in states, itertools.cycle([go.WHITE, go.BLACK]):
			won_game_list.append(st.get_winner() == game_color)
	# Concatenate tensors across turns within each game
	for i in xrange(mini_batch_size):
		X_list[i] = np.concatenate(X_list[i], axis=0)
		y_list[i] = np.vstack(y_list[i])
	return X_list, y_list, won_game_list


def train_batch(player, X_list, y_list, won_game_list, lr):
	"""Given the outcomes of a mini-batch of play against a fixed opponent,
		update the weights with reinforcement learning.

		Args:
		player -- player object with policy weights to be updated
		X_list -- List of one-hot encoded states.
		y_list -- List of one-hot encoded actions (to pair with X_list).
		winners -- List of winners corresponding to each item in
					training_pairs_list
		lr -- Keras learning rate

		Return:
		player -- same player, with updated weights.
	"""

	for X, y, won_game in zip(X_list, y_list, won_game_list):
		# Update weights in + direction if player won, and - direction if player lost.
		# Setting learning rate negative is hack for negative weights update.
		if won_game:
			player.policy.model.optimizer.lr.set_value(lr)
		else:
			player.policy.model.optimizer.lr.set_value(-lr)
		player.policy.model.fit(X, y, nb_epoch=1, batch_size=len(X))


def run_training(cmd_line_args=None):
	parser = argparse.ArgumentParser(description='Perform reinforcement learning to improve given policy network. Second phase of pipeline.')
	parser.add_argument("model_json", help="Path to policy model JSON.")
	parser.add_argument("initial_weights", help="Path to HDF5 file with inital weights (i.e. result of supervised training).")
	parser.add_argument("out_directory", help="Path to folder where the model params and metadata will be saved after each epoch.")
	parser.add_argument("--learning-rate", help="Keras learning rate (Default: .03)", type=float, default=.03)
	parser.add_argument("--policy-temp", help="Distribution temperature of players using policies (Default: 0.67)", type=float, default=0.67)
	parser.add_argument("--save-every", help="Save policy as a new opponent every n batches (Default: 500)", type=int, default=500)
	parser.add_argument("--game-batch", help="Number of games per mini-batch (Default: 20)", type=int, default=20)
	parser.add_argument("--iterations", help="Number of training batches/iterations (Default: 10000)", type=int, default=10000)
	parser.add_argument("--resume", help="Load latest weights in out_directory and resume", default=False, action="store_true")
	parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")
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
			print "copied {} to {}".format(args.initial_weights, os.path.join(args.out_directory, ZEROTH_FILE))
		player_weights = ZEROTH_FILE
	else:
		# if resuming, we expect initial_weights to be just a "weights.#####.hdf5" file, not a full path
		args.initial_weights = os.path.join(args.out_directory, os.path.basename(args.initial_weights))
		if not os.path.exists(args.initial_weights):
			raise ValueError("Cannot resume; weights {} do not exist".format(args.initial_weights))
		elif args.verbose:
			print "Resuming with weights {}".format(args.initial_weights)
		player_weights = os.path.basename(args.initial_weights)

	# Set initial conditions
	policy = CNNPolicy.load_model(args.model_json)
	policy.model.load_weights(args.initial_weights)
	player = ProbabilisticPolicyPlayer(policy, temperature=args.policy_temp)
	features = policy.preprocessor.feature_list

	# different opponents come from simply changing the weights of
	# opponent.policy.model "behind the scenes"
	opp_policy = CNNPolicy.load_model(args.model_json)
	opponent = ProbabilisticPolicyPlayer(opp_policy, temperature=args.policy_temp)

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
			"win_ratio": {}  # map from player to tuple of (opponent, win ratio) Useful for validating in lieu of 'accuracy/loss'
		}
	else:
		with open(os.path.join(args.out_directory, "metadata.json"), "r") as f:
			metadata = json.load(f)

	def save_metadata():
		with open(os.path.join(args.out_directory, "metadata.json"), "w") as f:
			json.dump(metadata, f)

	# Set SGD and compile
	sgd = SGD(lr=args.learning_rate)
	player.policy.model.compile(loss='binary_crossentropy', optimizer=sgd)
	board_size = player.policy.model.input_shape[-1]
	for i_iter in xrange(1, args.iterations + 1):
		# Train mini-batches by randomly choosing opponent from pool (possibly self)
		# and playing game_batch games against them
		opp_weights = np.random.choice(metadata["opponents"])
		opp_path = os.path.join(args.out_directory, opp_weights)
		# load new weights into opponent, but otherwise its the same
		opponent.policy.model.load_weights(opp_path)
		if args.verbose:
			print "Batch {}\tsampled opponent is {}".format(i_iter, opp_weights)
		# Make training pairs and do RL
		X_list, y_list, won_game_list = make_training_pairs(player, opponent, features, args.game_batch, board_size)
		win_ratio = np.sum(np.array(won_game_list) == True) / float(args.game_batch)
		metadata["win_ratio"][player_weights] = (opp_weights, win_ratio)
		train_batch(player, X_list, y_list, won_game_list, args.learning_rate)
		# Save intermediate models
		player_weights = "weights.%05d.hdf5" % i_iter
		player.policy.model.save_weights(os.path.join(args.out_directory, player_weights))
		# add player to batch of oppenents once in a while
		if i_iter % args.save_every == 0:
			metadata["opponents"].append(player_weights)
		save_metadata()

if __name__ == '__main__':
	run_training()
