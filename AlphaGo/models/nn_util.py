from keras import backend as K
from keras.models import model_from_json
from keras.engine.topology import Layer
from AlphaGo.preprocessing.preprocessing import Preprocess
import json


class NeuralNetBase(object):
    """Base class for neural network classes handling feature processing, construction
    of a 'forward' function, etc.
    """

    # keep track of subclasses to make generic saving/loading cleaner.
    # subclasses can be 'registered' with the @neuralnet decorator
    subclasses = {}

    def __init__(self, feature_list, **kwargs):
        """create a neural net object that preprocesses according to feature_list and uses
        a neural network specified by keyword arguments (using subclass' create_network())

        optional argument: init_network (boolean). If set to False, skips initializing
        self.model and self.forward and the calling function should set them.
        """
        self.preprocessor = Preprocess(feature_list)
        kwargs["input_dim"] = self.preprocessor.output_dim

        if kwargs.get('init_network', True):
            # self.__class__ refers to the subclass so that subclasses only
            # need to override create_network()
            self.model = self.__class__.create_network(**kwargs)
            # self.forward is a lambda function wrapping a Keras function
            self.forward = self._model_forward()

    def _model_forward(self):
        """Construct a function using the current keras backend that, when given a batch
        of inputs, simply processes them forward and returns the output

        This is as opposed to model.compile(), which takes a loss function
        and training method.

        c.f. https://github.com/fchollet/keras/issues/1426
        """
        # The uses_learning_phase property is True if the model contains layers that behave
        # differently during training and testing, e.g. Dropout or BatchNormalization.
        # In these cases, K.learning_phase() is a reference to a backend variable that should
        # be set to 0 when using the network in prediction mode and is automatically set to 1
        # during training.
        if self.model.uses_learning_phase:
            forward_function = K.function([self.model.input, K.learning_phase()],
                                          [self.model.output])

            # the forward_function returns a list of tensors
            # the first [0] gets the front tensor.
            return lambda inpt: forward_function([inpt, 0])[0]
        else:
            # identical but without a second input argument for the learning phase
            forward_function = K.function([self.model.input], [self.model.output])
            return lambda inpt: forward_function([inpt])[0]

    @staticmethod
    def load_model(json_file):
        """create a new neural net object from the architecture specified in json_file
        """
        with open(json_file, 'r') as f:
            object_specs = json.load(f)

        # Create object; may be a subclass of networks saved in specs['class']
        class_name = object_specs.get('class', 'CNNPolicy')
        try:
            network_class = NeuralNetBase.subclasses[class_name]
        except KeyError:
            raise ValueError("Unknown neural network type in json file: {}\n"
                             "(was it registered with the @neuralnet decorator?)"
                             .format(class_name))

        # create new object
        new_net = network_class(object_specs['feature_list'], init_network=False)

        new_net.model = model_from_json(object_specs['keras_model'], custom_objects={'Bias': Bias})
        if 'weights_file' in object_specs:
            new_net.model.load_weights(object_specs['weights_file'])
        new_net.forward = new_net._model_forward()
        return new_net

    def save_model(self, json_file, weights_file=None):
        """write the network model and preprocessing features to the specified file

        If a weights_file (.hdf5 extension) is also specified, model weights are also
        saved to that file and will be reloaded automatically in a call to load_model
        """
        # this looks odd because we are serializing a model with json as a string
        # then making that the value of an object which is then serialized as
        # json again.
        # It's not as crazy as it looks. A Network has 2 moving parts - the
        # feature preprocessing and the neural net, each of which gets a top-level
        # entry in the saved file. Keras just happens to serialize models with JSON
        # as well. Note how this format makes load_model fairly clean as well.
        object_specs = {
            'class': self.__class__.__name__,
            'keras_model': self.model.to_json(),
            'feature_list': self.preprocessor.feature_list
        }
        if weights_file is not None:
            self.model.save_weights(weights_file)
            object_specs['weights_file'] = weights_file
        # use the json module to write object_specs to file
        with open(json_file, 'w') as f:
            json.dump(object_specs, f)


def neuralnet(cls):
    """Class decorator for registering subclasses of NeuralNetBase
    """
    NeuralNetBase.subclasses[cls.__name__] = cls
    return cls


class Bias(Layer):
    """Custom keras layer that simply adds a scalar bias to each location in the input

    Largely copied from the keras docs:
    http://keras.io/layers/writing-your-own-keras-layers/#writing-your-own-keras-layers
    """

    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = K.zeros(input_shape[1:])
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x + self.W
