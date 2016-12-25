from keras.layers import convolutional, Dense
from keras.layers.core import Flatten
from keras.models import Sequential
from nn_util import NeuralNetBase, neuralnet


@neuralnet
class CNNValue(NeuralNetBase):
    """A convolutional neural network to guess the reward at the end of the
    game for a given board state, under the optimal policy.
    """

    def eval_state(self, state):
        """Given a GameState object, returns a value
        """
        tensor = self.preprocessor.state_to_tensor(state)
        # run the tensor through the network
        network_output = self.forward(tensor)
        return network_output[0]

    @staticmethod
    def create_network(**kwargs):
        """construct a convolutional neural network.
           Keword Arguments:
           - input_dim:            depth of features to be processed by first layer (no default)
           - board:                width of the go board to be processed (default 19)
           - filters_per_layer:    number of filters used on every layer (default 128)
           - layers:               number of convolutional steps (default 12)
           - filter_width_K:       (where K is between 1 and <layers>) width of filter on
                                   layer K (default 3 except 1st layer which defaults to 5).
                                   Must be odd.
        """

        defaults = {
            "board": 19,
            "filters_per_layer": 128,
            "layers": 13,  # layers 2-12 are identical to policy net
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

        # create all other layers (by default, this creates layers 2 through 12)
        # TODO: penultimate layer different in some way?
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

        # the last layer maps each <filters_per_layer> feature to a number
        network.add(convolutional.Convolution2D(
            nb_filter=1,
            nb_row=1,
            nb_col=1,
            init='uniform',
            activation='relu',
            border_mode='same'))
        network.add(Flatten())
        network.add(Dense(256, init='uniform', activation='relu'))
        network.add(Dense(1, init='uniform', activation="tanh"))
        return network
