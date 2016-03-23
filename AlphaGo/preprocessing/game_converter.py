import numpy as np
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import sgf_iter_states
import AlphaGo.go as go
import os

class game_converter:

	def _encode_label(self, move, size):
		"""Convert move into training label - a board with the given move
		marked as '1' and all other positions '0'
		"""
		one_hot = np.zeros((1, size, size),dtype=bool)
		(x,y) = move
		one_hot[0,x,y] = 1
		return one_hot

	def convert_game(self, file_name, features=None):
		"""Read the given SGF file into an iterable of (input,output) pairs
		for neural networks

		Each input is a GameState converted into one-hot neural net features
		Each output is an action encoded as a categorical choice from positions
		(see _encode_label)
		"""

		with open(file_name,'r') as file_object:
			state_action_iterator = sgf_iter_states(file_object.read())

		if features is None:
			proc = Preprocess()
		else:
			proc = Preprocess(features)

		for (state, move, player) in state_action_iterator:
			if move != go.PASS_MOVE:
				nn_input = proc.state_to_tensor(state)
				nn_output = self._encode_label(move, state.size)
				yield (nn_input, nn_output)

	def batch_convert(self, folder_path, features=None):
		"""lazily convert folder of games into training samples
		"""
		file_names = os.listdir(folder_path)
		for file_name in file_names:
			if file_name[-4:] != '.sgf': continue
			print file_name
			training_samples = self.convert_game(os.path.join(folder_path,file_name), features)
			for sample in training_samples:
				yield sample

if __name__ == '__main__':
	import argparse
	import json
	import cPickle as pickle

	parser = argparse.ArgumentParser(description='Prepare a folder of Go game files for training our neural network model.')
	parser.add_argument("--features", help="Comma-separated list of features to compute and store", default="board,ones,turns_since")
	parser.add_argument("--no-meta", help="Flag to disable saving a metadata.json file in the outfolder", default=False, action="store_true")
	parser.add_argument("infolder", help="Path to folder containing games")
	parser.add_argument("outfolder", help="Path to target folder.")
	parser.add_argument("-auto_split", help="Randomly place each sample into a train, test, or dev subfolder with probabilities .93, .05, and .02 respectively.",
	 					default=True)

	args = parser.parse_args()

	feature_list = args.features.split(",")

	if not args.no_meta:
		metafile = os.path.join(args.outfolder, "metadata.json")
		metadata = {
			"dataset": args.infolder,
			"features": feature_list
		}
		with open(metafile, "w") as f:
			json.dump(metadata, f)

	converter = game_converter()
	file_num = 0

	if args.auto_split:
		train_path = os.path.join(args.outfolder,'train')
		test_path = os.path.join(args.outfolder,'test')
		dev_path = os.path.join(args.outfolder,'dev')
		# create these subfolders if they don't exist
		if not os.path.exists(train_path): os.makedirs(train_path)
		if not os.path.exists(test_path): os.makedirs(test_path)
		if not os.path.exists(dev_path): os.makedirs(dev_path)
	for s_a_tuple in converter.batch_convert(args.infolder, features=feature_list):
		file_name = str(file_num)+".pkl"
		if args.auto_split:
			# select subfolder randomly w/ appropriate probability
			subfolder = np.random.choice([train_path,test_path,dev_path], 1, p=[.93,.05,.02])[0]
			pickle.dump(s_a_tuple, open(os.path.join(subfolder,file_name), "wb"))
		else:
			pickle.dump(s_a_tuple, open(os.path.join(args.outfolder,file_name), "wb"))
		file_num += 1
