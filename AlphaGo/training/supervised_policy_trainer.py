import numpy as np
import os
import h5py as h5
import json
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess


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
            # get rotation symmetry belonging to state
            transform = transforms[data_idx[1]]
            # get state from dataset and transform it.
            # loop comprehension is used so that the transformation acts on the
            # 3rd and 4th dimensions
            state = np.array([transform(plane) for plane in state_dataset[data_idx[0]]])
            # must be cast to a tuple so that it is interpreted as (x,y) not [(x,:), (y,:)]
            action_xy = tuple(action_dataset[data_idx[0]])
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


TRANSFORMATION_INDICES = {
    "noop": 0,
    "rot90": 1,
    "rot180": 2,
    "rot270": 3,
    "fliplr": 4,
    "flipud": 5,
    "diag1": 6,
    "diag2": 7
}

BOARD_TRANSFORMATIONS = {
    0: lambda feature: feature,
    1: lambda feature: np.rot90(feature, 1),
    2: lambda feature: np.rot90(feature, 2),
    3: lambda feature: np.rot90(feature, 3),
    4: lambda feature: np.fliplr(feature),
    5: lambda feature: np.flipud(feature),
    6: lambda feature: np.transpose(feature),
    7: lambda feature: np.fliplr(np.rot90(feature, 1))
}


def load_indices_from_file(shuffle_file):
    # load indices from shuffle_file
    with open(shuffle_file, "r") as f:
        indices = np.load(f)

    return indices


def save_indices_to_file(shuffle_file, indices):
    # save indices to shuffle_file
    with open(shuffle_file, "w") as f:
        np.save(f, indices)


def create_and_save_shuffle_indices(minibatch_size, train_val_test, max_validation,
                                    n_total_data_size, symmetries, shuffle_file_train,
                                    shuffle_file_val, shuffle_file_test):
    """ create an array with all unique state and symmetry pairs,
        calculate test/validation/training set sizes,
        seperate those sets and save them to seperate files.
    """

    # Create an array with a unique row for each combination of a training example
    # and a symmetry.
    # shuffle_indices[i][0] is an index into training examples,
    # shuffle_indices[i][1] is the index (from 0 to 7) of the symmetry transformation to apply
    shuffle_indices = np.empty(shape=[n_total_data_size * len(symmetries), 2], dtype=int)
    for dataset_idx in range(n_total_data_size):
        for symmetry_idx in range(len(symmetries)):
            shuffle_indices[dataset_idx * len(symmetries) + symmetry_idx][0] = dataset_idx
            shuffle_indices[dataset_idx * len(symmetries) +
                            symmetry_idx][1] = symmetries[symmetry_idx]

    # shuffle rows without affecting x,y pairs
    np.random.shuffle(shuffle_indices)

    # validation set size
    n_val_data = int(train_val_test[1] * len(shuffle_indices))
    # limit validation set to --max-validation
    if n_val_data > max_validation:
        n_val_data = max_validation

    # test set size
    n_test_data = int(train_val_test[2] * len(shuffle_indices))

    # train set size
    n_train_data = len(shuffle_indices) - n_val_data - n_test_data

    # Need to make sure training data is divisible by minibatch size or get
    # warning mentioning accuracy from keras
    remainder = n_train_data % minibatch_size
    n_train_data -= remainder
    n_test_data += remainder

    # create training set and save
    train_indices = shuffle_indices[0:n_train_data]
    save_indices_to_file(shuffle_file_train, train_indices)

    # create validation set and save
    val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
    save_indices_to_file(shuffle_file_val, val_indices)

    # create test set and save
    test_indices = shuffle_indices[n_train_data + n_val_data:
                                   n_train_data + n_val_data + n_test_data]
    save_indices_to_file(shuffle_file_test, test_indices)

    return train_indices, val_indices, test_indices


def get_train_val_test_indices(args, meta_writer, resume, dataset):

    # used symmetries
    if args.symmetries == "all":
        # add all symmetries
        symmetries = [TRANSFORMATION_INDICES[name] for name in TRANSFORMATION_INDICES]
    elif args.symmetries == "none":
        # only add standart orientation
        symmetries = [TRANSFORMATION_INDICES["noop"]]
    else:
        # add specified symmetries
        symmetries = [TRANSFORMATION_INDICES[name] for name in args.symmetries.strip().split(",")]

    # shuffle file locations for train/validation/test set
    shuffle_file_train = os.path.join(args.out_directory, "shuffle_train.npz")
    shuffle_file_val = os.path.join(args.out_directory, "shuffle_validate.npz")
    shuffle_file_test = os.path.join(args.out_directory, "shuffle_test.npz")

    # check train/validation/test shuffle file existence, resume,
    # train_data file is the same and amount of states is equal
    if resume and os.path.exists(shuffle_file_train) and os.path.exists(shuffle_file_val) and \
       os.path.exists(shuffle_file_test) and \
       meta_writer.metadata["training_data"] == args.train_data and \
       meta_writer.metadata["available_states"] == len(dataset["states"]):
        # load from .npz files
        train_indices = load_indices_from_file(shuffle_file_train)
        val_indices = load_indices_from_file(shuffle_file_val)
        test_indices = load_indices_from_file(shuffle_file_test)

        if args.verbose:
            print("loading previous data shuffling indices")
    else:
        # shuffle data, add rotations, save rotations to files
        train_indices, val_indices, test_indices = create_and_save_shuffle_indices(
                args.minibatch, args.train_val_test, args.max_validation, len(dataset["states"]),
                symmetries, shuffle_file_train, shuffle_file_val, shuffle_file_test)

        # save amount of states to metadata
        meta_writer.metadata["available_states"] = len(dataset["states"])
        # save training data file to metadata
        meta_writer.metadata["training_data"] = args.train_data

        if args.verbose:
            print("created new data shuffling indices")

    if args.verbose:
        print("dataset loaded")
        print("\t%d total positions" % len(dataset["states"]))
        print("\t%d total samples" % (len(dataset["states"]) * len(symmetries)))
        print("\t%d total samples check" % (len(train_indices) +
              len(val_indices) + len(test_indices)))
        print("\t%d training samples" % len(train_indices))
        print("\t%d validation samples" % len(val_indices))
        print("\t%d test samples" % len(test_indices))

    return train_indices, val_indices, test_indices


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
    parser.add_argument("--train-val-test", help="Fraction of data to use for training/val/test. Must sum to 1. Invalid if restarting training", nargs=3, type=float, default=[0.95, .05, .0])  # noqa: E501
    parser.add_argument("--max-validation", help="maximum validation set size", type=int, default=5000000)  # noqa: E501
    parser.add_argument("--symmetries", help="none, all or comma-separated list of transforms, subset of noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2", default='all')  # noqa: E501
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
                print("starting fresh output directory %s" % args.out_directory)

    # load model from json spec
    policy = CNNPolicy.load_model(args.model)
    model_features = policy.preprocessor.feature_list
    model = policy.model
    if resume:
        model.load_weights(os.path.join(args.out_directory, args.weights))

    # features of training data
    dataset = h5.File(args.train_data)

    # Verify that dataset's features match the model's expected features.
    if 'features' in dataset:
        dataset_features = dataset['features'][()]
        dataset_features = dataset_features.split(",")
        if len(dataset_features) != len(model_features) or \
           any(df != mf for (df, mf) in zip(dataset_features, model_features)):
            raise ValueError("Model JSON file expects features \n\t%s\n"
                             "But dataset contains \n\t%s" % ("\n\t".join(model_features),
                                                              "\n\t".join(dataset_features)))
        elif args.verbose:
            print("Verified that dataset features and model features exactly match.")
    else:
        # Cannot check each feature, but can check number of planes.
        n_dataset_planes = dataset["states"].shape[1]
        tmp_preprocess = Preprocess(model_features)
        n_model_planes = tmp_preprocess.output_dim
        if n_dataset_planes != n_model_planes:
            raise ValueError("Model JSON file expects a total of %d planes from features \n\t%s\n"
                             "But dataset contains %d planes" % (n_model_planes,
                                                                 "\n\t".join(model_features),
                                                                 n_dataset_planes))
        elif args.verbose:
            print("Verified agreement of number of model and dataset feature planes, but cannot "
                  "verify exact match using old dataset format.")

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
        print("starting with empty metadata")
    # the MetadataWriterCallback only sets 'epoch' and 'best_epoch'. We can add
    # in anything else we like here
    #
    # TODO - model and train_data are saved in meta_file; check that they match
    # (and make args optional when restarting?)
    meta_writer.metadata["model_file"] = args.model
    # Record all command line args in a list so that all args are recorded even
    # when training is stopped and resumed.

    # TODO find out why the commented line gives error after restart:
    # AttributeError: 'NoneType' object has no attribute 'append'
    meta_args_data = meta_writer.metadata.get("cmd_line_args", [])
    meta_args_data.append(vars(args))
    meta_writer.metadata["cmd_line_args"] = meta_args_data
    # meta_writer.metadata["cmd_line_args"] \
    #    = meta_writer.metadata.get("cmd_line_args", []).append(vars(args))

    # create ModelCheckpoint to save weights every epoch
    checkpoint_template = os.path.join(args.out_directory, "weights.{epoch:05d}.hdf5")
    checkpointer = ModelCheckpoint(checkpoint_template)

    # get train/validation/test indices
    train_indices, val_indices, test_indices \
        = get_train_val_test_indices(args, meta_writer, resume, dataset)

    # create dataset generators
    train_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        train_indices,
        args.minibatch,
        BOARD_TRANSFORMATIONS)
    val_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        val_indices,
        args.minibatch,
        BOARD_TRANSFORMATIONS)

    sgd = SGD(lr=args.learning_rate, decay=args.decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    samples_per_epoch = args.epoch_length or len(train_indices)

    if args.verbose:
        print("STARTING TRAINING")

    model.fit_generator(
        generator=train_data_generator,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=args.epochs,
        callbacks=[checkpointer, meta_writer],
        validation_data=val_data_generator,
        nb_val_samples=len(val_indices))


if __name__ == '__main__':
    run_training()
