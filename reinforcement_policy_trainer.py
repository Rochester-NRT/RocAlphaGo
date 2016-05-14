import argparse
from keras.callbacks import ModelCheckpoint
from AlphaGo.ai import ProbabilisticPolicyPlayer
import AlphaGo.go as go
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import flatten_idx
from keras.optimizers import SGD
import json
import numpy as np
# np.set_printoptions(linewidth=160)
import os
#
# from ipdb import set_trace as BP


def make_training_pairs(player, opp, features, mini_batch_size):
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

	def do_move(states, states_prev, moves, X_list, y_list, player_color):
		bsize_flat = bsize * bsize
		for st, st_prev, mv, X, y in zip(states, states_prev, moves, X_list,
										 y_list):
			if not st.is_end_of_game:
				# Only do more moves if not end of game already
				st.do_move(mv)
				if st.current_player != player_color and mv is not go.PASS_MOVE:
					# Convert move to one-hot
					state_1hot = preprocessor.state_to_tensor(st_prev)
					move_1hot = np.zeros(bsize_flat)
					move_1hot[flatten_idx(mv, bsize)] = 1
					X.append(state_1hot)
					y.append(move_1hot)
		return states, X_list, y_list

	# Lists of game training pairs (1-hot)
	X_list = [list() for _ in xrange(mini_batch_size)]
	y_list = [list() for _ in xrange(mini_batch_size)]
	preprocessor = Preprocess(features)
	bsize = player.policy.model.input_shape[-1]
	states = [GameState() for i in xrange(mini_batch_size)]
	# Randomly choose who goes first (i.e. color of 'player')
	player_color = np.random.choice([go.BLACK, go.WHITE])
	player1, player2 = (player, opp) if player_color == go.BLACK else \
		(opp, player)
	while True:
		# Cache states before moves
		states_prev = [st.copy() for st in states]
		# Get moves (batch)
		moves_black = player1.get_moves(states)
		# Do moves (black)
		states, X_list, y_list = do_move(states, states_prev, moves_black,
										 X_list, y_list, player_color)
		# Do moves (white)
		moves_white = player2.get_moves(states)
		states, X_list, y_list = do_move(states, states_prev, moves_white,
										 X_list, y_list, player_color)
		# If all games have ended, we're done. Get winners.
		done = [st.is_end_of_game for st in states]
		if all(done):
			break
	winners = [st.get_winner() for st in states]
	# Concatenate tensors across turns within each game
	for i in xrange(mini_batch_size):
		X_list[i] = np.concatenate(X_list[i], axis=0)
		y_list[i] = np.vstack(y_list[i])
	return X_list, y_list, winners


def train_batch(player, X_list, y_list, winners, lr):
	"""Given the outcomes of a mini-batch of play against a fixed opponent,
	   update the weights with reinforcement learning in place.

	   Args:
	   player -- player object with policy weights to be updated
	   X_list -- List of one-hot encoded states.
	   y_list -- List of one-hot encoded actions (to pair with X_list).
	   winners -- List of winners corresponding to each item in
				  training_pairs_list
	   lr -- Keras learning rate
	"""

	for X, y, winner in zip(X_list, y_list, winners):
		# Update weights in + direction if player won, and - direction if player lost.
		# Setting learning rate negative is hack for negative weights update.
		if winner == -1:
			player.policy.model.optimizer.lr.set_value(-lr)
		else:
			player.policy.model.optimizer.lr.set_value(lr)
		player.policy.model.fit(X, y, nb_epoch=1, batch_size=len(X))


def run(player, args, opponents, features, model_folder):
	# Set SGD and compile
	sgd = SGD(lr=args.learning_rate)
	player.policy.model.compile(loss='binary_crossentropy', optimizer=sgd)
	player_wins_per_batch = []
	kwargs = {'filters_per_layer': 128, 'layers': 12} # args to CNNPolicy.create_network()
	preprocessor = Preprocess(features)
	kwargs["input_dim"] = preprocessor.output_dim
	for i_iter in xrange(args.iterations):
		# Train mini-batches
		for i_batch in xrange(args.save_every):
			# Randomly choose opponent from pool
			opp_filepath = np.random.choice(os.listdir(model_folder))
			opp_path = os.path.join(model_folder,opp_filepath)
			oppPolicy = CNNPolicy(features, **kwargs);
			oppPolicy.model.load_weights(opp_path)
			opp = ProbabilisticPolicyPlayer(oppPolicy)
			# Make training pairs and do RL
			X_list, y_list, winners = make_training_pairs(
				player, opp, features, args.game_batch_size)
			n_wins = np.sum(np.array(winners) == 1)
			player_wins_per_batch.append(n_wins)
			print 'Number of wins this batch: {}/{}'.format(n_wins, args.game_batch_size)
			train_batch(player, X_list, y_list, winners, args.learning_rate)
		# Save intermediate models
		model_path = os.path.join(model_folder,str(i_iter))
		player.policy.model.save_weights(model_path)

def get_policy(weights_file, metapath):
	with open(metapath) as metafile:
	    metadata = json.load(metafile)
	arch = {'filters_per_layer': 128, 'layers': 12} # args to CNNPolicy.create_network()
	policy = CNNPolicy(feature_list=metadata['feature_list'], **arch);
	policy.model.load_weights(weights_file);
	policy.model.compile(loss='categorical_crossentropy', optimizer='sgd')
	return policy


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Perform reinforcement learning '
									 'to improve given policy network. Second phase of pipeline.')
	parser.add_argument("--initial_weights", help="Path to file with weights to start from.", default="D:\\ps\\club\\Go\\models\weights.00008.hdf5")
	parser.add_argument("--initial_json", help="Path to file with initial network params.", default="D:\\ps\\club\\Go\\all_feat_model.json")
	parser.add_argument("--model_folder", help="Path to folder where the model "
						"params will be saved after each epoch. Default: None", default="d:\\ps\\club\\Go\\exp\\models")
	parser.add_argument(
		"--learning_rate", help="Keras learning rate (Default: .03)", type=float,
		default=.03)
	parser.add_argument(
		"--save_every", help="Save policy every n mini-batches (Default: 500)",
		type=int, default=500)
	parser.add_argument(
		"--game_batch_size", help="Number of games per mini-batch (Default: 20)",
		type=int, default=20)
	parser.add_argument(
		"--iterations", help="Number of training iterations (i.e. mini-batch) "
		"(Default: 20)",
		type=int, default=20)
	# Baseline function (TODO) default lambda state: 0  (receives either file
	# paths to JSON and weights or None, in which case it uses default baseline 0)
	args = parser.parse_args()
	# Load policy from file
	# policy = model_from_json(open(args.initial_json).read())
	# policy.load_weights(args.initial_weights)
	# player = ProbabilisticPolicyPlayer(model)
	#############################################
	# Just for now, while we get the model directories set up...
	features = ["board", "ones", "turns_since", "liberties", "capture_size",
				"self_atari_size", "liberties_after",
				"sensibleness", "zeros"]
	#policy = CNNPolicy(features)
	policy = get_policy(args.initial_weights, args.initial_json)
	player = ProbabilisticPolicyPlayer(policy)
	#############################################
	# Load opponent pool
	opponents = []
	if args.model_folder is not None:
		# TODO
		opponent_files = next(os.walk(args.model_folder))[2]
		if len(args.model_folder) == 0:
			# No opponents yet, so play against self
			opponents = [player]
		else:
			# TODO
			opponents = [player]
	else:
		opponents = [player]
	opponents = run(player, args, opponents, features, args.model_folder)
