from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import convolutional
from keras.layers.core import Activation, Flatten
import keras.backend as K
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import flatten_idx
import json


class CNNPolicy(object):
	"""uses a convolutional neural network to evaluate the state of the game
	and compute a probability distribution over the next action
	"""

	def __init__(self, feature_list, **kwargs):
		"""create a policy object that preprocesses according to feature_list and uses
		a neural network specified by keyword arguments (see create_network())
		"""
		self.preprocessor = Preprocess(feature_list)
		kwargs["input_dim"] = self.preprocessor.output_dim
		self.model = CNNPolicy.create_network(**kwargs)
		self.forward = self._model_forward()

	def _model_forward(self):
		"""Construct a function using the current keras backend that, when given a batch
		of inputs, simply processes them forward and returns the output

		The output has size (batch x 361) for 19x19 boards (i.e. the output is a batch
		of distributions over flattened boards. See AlphaGo.util#flatten_idx)

		This is as opposed to model.compile(), which takes a loss function
		and training method.

		c.f. https://github.com/fchollet/keras/issues/1426
		"""
		forward_function = K.function([self.model.input], [self.model.output])

		# the forward_function returns a list of tensors
		# the first [0] gets the front tensor.
		return lambda inpt: forward_function([inpt])[0]

	def batch_eval_state(self, state_gen, batch=16):
		"""Given a stream of states in state_gen, evaluates them in batches
		to make best use of GPU resources.

		Returns: TBD (stream of results? that would break zip().
			streaming pairs of pre-zipped (state, result)?)
		"""
		raise NotImplementedError()

	def eval_state(self, state, moves=None):
		"""Given a GameState object, returns a list of (action, probability) pairs
		according to the network outputs

		If a list of moves is specified, only those moves are kept in the distribution
		"""
		tensor = self.preprocessor.state_to_tensor(state)

		# run the tensor through the network
		network_output = self.forward(tensor)

		moves = moves or state.get_legal_moves()
		move_indices = [flatten_idx(m, state.size) for m in moves]

		# get network activations at legal move locations
		# note: may not be a proper distribution by ignoring illegal moves
		distribution = network_output[0][move_indices]
		distribution = distribution / distribution.sum()
		return zip(moves, distribution)

	@staticmethod
	def create_network(**kwargs):
		"""construct a convolutional neural network.

		Keword Arguments:
		- input_dim:         	depth of features to be processed by first layer (no default)
		- board:             	width of the go board to be processed (default 19)
		- filters_per_layer: 	number of filters used on every layer (default 128)
		- layers:            	number of convolutional steps (default 12)
		- filter_width_K:    	(where K is between 1 and <layers>) width of filter on
								layer K (default 3 except 1st layer which defaults to 5).
								Must be odd.
		"""
		defaults = {
			"board": 19,
			"filters_per_layer": 128,
			"layers": 12,
			"filter_width_1": 5
		}
		# copy defaults, but override with anything in kwargs
		params = defaults
		params.update(kwargs)

		# create the network:
		# a series of zero-paddings followed by convolutions
		# such that the output dimensions are also board x board
		network = Sequential()

		# create first layer
		network.add(convolutional.Convolution2D(
			input_shape=(params["input_dim"], params["board"], params["board"]),
			nb_filter=params["filters_per_layer"],
			nb_row=params["filter_width_1"],
			nb_col=params["filter_width_1"],
			init='uniform',
			activation='relu',
			border_mode='same'))

		# create all other layers
		for i in range(2, params["layers"] + 1):
			# use filter_width_K if it is there, otherwise use 3
			filter_key = "filter_width_%d" % i
			filter_width = params.get(filter_key, 3)
			network.add(convolutional.Convolution2D(
				nb_filter=params["filters_per_layer"],
				nb_row=filter_width,
				nb_col=filter_width,
				init='uniform',
				activation='relu',
				border_mode='same'))

		# the last layer maps each <filters_per_layer> featuer to a number
		network.add(convolutional.Convolution2D(
			nb_filter=1,
			nb_row=1,
			nb_col=1,
			init='uniform',
			border_mode='same'))
		# reshape output to be board x board
		network.add(Flatten())
		# softmax makes it into a probability distribution
		network.add(Activation('softmax'))

		return network

	@staticmethod
	def load_model(json_file):
		"""create a new CNNPolicy object from the architecture specified in json_file
		"""
		with open(json_file, 'r') as f:
			object_specs = json.load(f)
		new_policy = CNNPolicy(object_specs['feature_list'])
		new_policy.model = model_from_json(object_specs['keras_model'])
		if 'weights_file' in object_specs:
			new_policy.model.load_weights(object_specs['weights_file'])
		new_policy.forward = new_policy._model_forward()
		return new_policy

	def save_model(self, json_file, weights_file=None):
		"""write the network model and preprocessing features to the specified file

		If a weights_file (.hdf5 extension) is also specified, model weights are also
		saved to that file and will be reloaded automatically in a call to load_model
		"""
		# this looks odd because we are serializing a model with json as a string
		# then making that the value of an object which is then serialized as
		# json again.
		# It's not as crazy as it looks. A CNNPolicy has 2 moving parts - the
		# feature preprocessing and the neural net, each of which gets a top-level
		# entry in the saved file. Keras just happens to serialize models with JSON
		# as well. Note how this format makes load_model fairly clean as well.
		object_specs = {
			'keras_model': self.model.to_json(),
			'feature_list': self.preprocessor.feature_list
		}
		if weights_file is not None:
			self.model.save_weights(weights_file)
			object_specs['weights_file'] = weights_file
		# use the json module to write object_specs to file
		with open(json_file, 'w') as f:
			json.dump(object_specs, f)
