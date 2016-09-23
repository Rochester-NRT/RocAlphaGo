import numpy as np
import os
import h5py as h5
import json
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback
from AlphaGo.models.policy import CNNPolicy


def one_hot_action(action, size=19):
    """Convert an (x,y) action into a size x size array of zeros with a 1 at x,y
    """
    categorical = np.zeros((size, size))
    categorical[action] = 1
    return categorical


def shuffled_hdf5_batch_generator(state_dataset, action_dataset,
                                  indices, batch_size, transforms=[]):
    """A generator of batches of training data for use with the fit_generator function
    of Keras. Data is accessed in the order of the given indices for shuffling.
    """
    state_batch_shape = (batch_size,) + state_dataset.shape[1:]
    game_size = state_batch_shape[-1]
    Xbatch = np.zeros(state_batch_shape)
    Ybatch = np.zeros((batch_size, game_size * game_size))
    batch_idx = 0
    while True:
        for data_idx in indices:
            # choose a random transformation of the data (rotations/reflections of the board)
            transform = np.random.choice(transforms)
            # get state from dataset and transform it.
            # loop comprehension is used so that the transformation acts on the
            # 3rd and 4th dimensions
            state = np.array([transform(plane) for plane in state_dataset[data_idx]])
            # must be cast to a tuple so that it is interpreted as (x,y) not [(x,:), (y,:)]
            action_xy = tuple(action_dataset[data_idx])
            action = transform(one_hot_action(action_xy, game_size))
            Xbatch[batch_idx] = state
            Ybatch[batch_idx] = action.flatten()
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                yield (Xbatch, Ybatch)


class MetadataWriterCallback(Callback):

    def __init__(self, path):
        self.file = path
        self.metadata = {
            "epochs": [],
            "best_epoch": 0
        }

    def on_epoch_end(self, epoch, logs={}):
        # in case appending to logs (resuming training), get epoch number ourselves
        epoch = len(self.metadata["epochs"])

        self.metadata["epochs"].append(logs)

        if "val_loss" in logs:
            key = "val_loss"
        else:
            key = "loss"

        best_loss = self.metadata["epochs"][self.metadata["best_epoch"]][key]
        if logs.get(key) < best_loss:
            self.metadata["best_epoch"] = epoch

        with open(self.file, "w") as f:
            json.dump(self.metadata, f, indent=2)


BOARD_TRANSFORMATIONS = {
    "noop": lambda feature: feature,
    "rot90": lambda feature: np.rot90(feature, 1),
    "rot180": lambda feature: np.rot90(feature, 2),
    "rot270": lambda feature: np.rot90(feature, 3),
    "fliplr": lambda feature: np.fliplr(feature),
    "flipud": lambda feature: np.flipud(feature),
    "diag1": lambda feature: np.transpose(feature),
    "diag2": lambda feature: np.fliplr(np.rot90(feature, 1))
}


def run_training(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list
    """
    import argparse
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    # required args
    parser.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicy.save_model())")  # noqa: E501
    parser.add_argument("train_data", help="A .h5 file of training data")
    parser.add_argument("out_directory", help="directory where metadata and weights will be saved")
    # frequently used args
    parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 16", type=int, default=16)  # noqa: E501
    parser.add_argument("--epochs", "-E", help="Total number of iterations on the data. Default: 10", type=int, default=10)  # noqa: E501
    parser.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)  # noqa: E501
    parser.add_argument("--learning-rate", "-r", help="Learning rate - how quickly the model learns at first. Default: .03", type=float, default=.03)  # noqa: E501
    parser.add_argument("--decay", "-d", help="The rate at which learning decreases. Default: .0001", type=float, default=.0001)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # slightly fancier args
    parser.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training", default=None)  # noqa: E501
    parser.add_argument("--train-val-test", help="Fraction of data to use for training/val/test. Must sum to 1. Invalid if restarting training", nargs=3, type=float, default=[0.93, .05, .02])  # noqa: E501
    parser.add_argument("--symmetries", help="Comma-separated list of transforms, subset of noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2", default='noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2')  # noqa: E501
    # TODO - an argument to specify which transformations to use, put it in metadata

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # TODO - what follows here should be refactored into a series of small functions

    resume = args.weights is not None

    if args.verbose:
        if resume:
            print("trying to resume from %s with weights %s" %
                  (args.out_directory, os.path.join(args.out_directory, args.weights)))
        else:
            if os.path.exists(args.out_directory):
                print("directory %s exists. any previous data will be overwritten" %
                      args.out_directory)
            else:
                print "starting fresh output directory %s" % args.out_directory

    # load model from json spec
    model = CNNPolicy.load_model(args.model).model
    if resume:
        model.load_weights(os.path.join(args.out_directory, args.weights))

    # TODO - (waiting on game_converter) verify that features of model match
    # features of training data
    dataset = h5.File(args.train_data)
    n_total_data = len(dataset["states"])
    n_train_data = int(args.train_val_test[0] * n_total_data)
    # Need to make sure training data is divisible by minibatch size or get
    # warning mentioning accuracy from keras
    n_train_data = n_train_data - (n_train_data % args.minibatch)
    n_val_data = n_total_data - n_train_data
    # n_test_data = n_total_data - (n_train_data + n_val_data)

    if args.verbose:
        print "datset loaded"
        print "\t%d total samples" % n_total_data
        print "\t%d training samples" % n_train_data
        print "\t%d validaion samples" % n_val_data

    # ensure output directory is available
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    # create metadata file and the callback object that will write to it
    meta_file = os.path.join(args.out_directory, "metadata.json")
    meta_writer = MetadataWriterCallback(meta_file)
    # load prior data if it already exists
    if os.path.exists(meta_file) and resume:
        with open(meta_file, "r") as f:
            meta_writer.metadata = json.load(f)
        if args.verbose:
            print("previous metadata loaded: %d epochs. new epochs will be appended." %
                  len(meta_writer.metadata["epochs"]))
    elif args.verbose:
        print "starting with empty metadata"
    # the MetadataWriterCallback only sets 'epoch' and 'best_epoch'. We can add
    # in anything else we like here
    #
    # TODO - model and train_data are saved in meta_file; check that they match
    # (and make args optional when restarting?)
    meta_writer.metadata["training_data"] = args.train_data
    meta_writer.metadata["model_file"] = args.model
    # Record all command line args in a list so that all args are recorded even
    # when training is stopped and resumed.
    meta_writer.metadata["cmd_line_args"] \
        = meta_writer.metadata.get("cmd_line_args", []).append(vars(args))

    # create ModelCheckpoint to save weights every epoch
    checkpoint_template = os.path.join(args.out_directory, "weights.{epoch:05d}.hdf5")
    checkpointer = ModelCheckpoint(checkpoint_template)

    # load precomputed random-shuffle indices or create them
    # TODO - save each train/val/test indices separately so there's no danger of
    # changing args.train_val_test when resuming
    shuffle_file = os.path.join(args.out_directory, "shuffle.npz")
    if os.path.exists(shuffle_file) and resume:
        with open(shuffle_file, "r") as f:
            shuffle_indices = np.load(f)
        if args.verbose:
            print "loading previous data shuffling indices"
    else:
        # create shuffled indices
        shuffle_indices = np.random.permutation(n_total_data)
        with open(shuffle_file, "w") as f:
            np.save(f, shuffle_indices)
        if args.verbose:
            print "created new data shuffling indices"
    # training indices are the first consecutive set of shuffled indices, val
    # next, then test gets the remainder
    train_indices = shuffle_indices[0:n_train_data]
    val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
    # test_indices = shuffle_indices[n_train_data + n_val_data:]

    symmetries = [BOARD_TRANSFORMATIONS[name] for name in args.symmetries.strip().split(",")]

    # create dataset generators
    train_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        train_indices,
        args.minibatch,
        symmetries)
    val_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        val_indices,
        args.minibatch,
        symmetries)

    sgd = SGD(lr=args.learning_rate, decay=args.decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    samples_per_epoch = args.epoch_length or n_train_data

    if args.verbose:
        print "STARTING TRAINING"

    model.fit_generator(
        generator=train_data_generator,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=args.epochs,
        callbacks=[checkpointer, meta_writer],
        validation_data=val_data_generator,
        nb_val_samples=n_val_data)

if __name__ == '__main__':
    run_training()
