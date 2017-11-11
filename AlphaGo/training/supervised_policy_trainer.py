import os
import json
import threading
import h5py as h5
import numpy as np
from keras import backend as K
from AlphaGo.util import confirm
from keras.optimizers import SGD
from keras.callbacks import Callback
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess

# default settings
DEFAULT_MAX_VALIDATION = 1000000000
DEFAULT_TRAIN_VAL_TEST = [.95, .05, .0]
DEFAULT_LEARNING_RATE = .003
DEFAULT_BATCH_SIZE = 16
DEFAULT_DECAY = .0000000125
DEFAULT_EPOCH = 10

# metdata file
FILE_METADATA = 'metadata_policy_supervised.json'
# weight folder
FOLDER_WEIGHT = os.path.join('policy_supervised_weights')

# shuffle files
FILE_VALIDATE = 'shuffle_policy_validate.npz'
FILE_TRAIN = 'shuffle_policy_train.npz'
FILE_TEST = 'shuffle_policy_test.npz'

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


def one_hot_action(action, size=19):
    """Convert an (x,y) action into a size x size array of zeros with a 1 at x,y
    """

    categorical = np.zeros((size, size))
    categorical[action] = 1
    return categorical


class threading_shuffled_hdf5_batch_generator:
    """A generator of batches of training data for use with the fit_generator function
       of Keras. Data is accessed in the order of the given indices for shuffling.

       it is threading safe but not multiprocessing therefore only use it with
       pickle_safe=False when using multiple workers
    """

    def shuffle_indices(self, seed=None, idx=0):
        # set generator_sample to idx
        self.metadata['generator_sample'] = idx

        # check if seed is provided or generate random
        if seed is None:
            # create random seed
            self.metadata['generator_seed'] = np.random.random_integers(4294967295)

        # feed numpy.random with seed in order to continue with certain batch
        np.random.seed(self.metadata['generator_seed'])
        # shuffle indices according to seed
        if not self.validation:
            np.random.shuffle(self.indices)

    def __init__(self, state_dataset, action_dataset, indices, batch_size, metadata=None,
                 validation=False):
        self.action_dataset = action_dataset
        self.state_dataset = state_dataset
        # lock used for multithreaded workers
        self.data_lock = threading.Lock()
        self.indices_max = len(indices)
        self.validation = validation
        self.batch_size = batch_size
        self.indices = indices

        if metadata is not None:
            self.metadata = metadata
        else:
            # create metadata object
            self.metadata = {
                "generator_seed": None,
                "generator_sample": 0
            }

        # shuffle indices
        # when restarting generator_seed and generator_batch will
        # reset generator to the same point as before
        self.shuffle_indices(self.metadata['generator_seed'], self.metadata['generator_sample'])

    def __iter__(self):
        return self

    def next_indice(self):
        # use lock to prevent double hdf5 acces and incorrect generator_sample increment
        with self.data_lock:

            # get next training sample
            training_sample = self.indices[self.metadata['generator_sample'], :]
            # get state
            state = self.state_dataset[training_sample[0]]
            # get action
            # must be cast to a tuple so that it is interpreted as (x,y) not [(x,:), (y,:)]
            action = tuple(self.action_dataset[training_sample[0]])

            # increment generator_sample
            self.metadata['generator_sample'] += 1
            # shuffle indices when all have been used
            if self.metadata['generator_sample'] >= self.indices_max:
                self.shuffle_indices()

            # return state, action and transformation
            return state, action, training_sample[1]

    def next(self):
        state_batch_shape = (self.batch_size,) + self.state_dataset.shape[1:]
        game_size = state_batch_shape[-1]
        Xbatch = np.zeros(state_batch_shape)
        Ybatch = np.zeros((self.batch_size, game_size * game_size))

        for batch_idx in xrange(self.batch_size):
            state, action, transformation = self.next_indice()

            # get rotation symmetry belonging to state
            transform = BOARD_TRANSFORMATIONS[transformation]

            # get state from dataset and transform it.
            # loop comprehension is used so that the transformation acts on the
            # 3rd and 4th dimensions
            state_transform = np.array([transform(plane) for plane in state])
            action_transform = transform(one_hot_action(action, game_size))

            Xbatch[batch_idx] = state_transform
            Ybatch[batch_idx] = action_transform.flatten()

        return (Xbatch, Ybatch)


class LrDecayCallback(Callback):
    """Set learning rate every batch according to:
       initial_learning_rate * (1. / (1. + self.decay * curent_batch))
    """

    def __init__(self, metadata):
        super(LrDecayCallback, self).__init__()
        self.learning_rate = metadata["learning_rate"]
        self.decay = metadata["decay"]
        self.metadata = metadata

    def set_lr(self):
        # calculate learning rate
        new_lr = self.learning_rate * (1. / (1. + self.decay * self.metadata["current_batch"]))

        # set new learning rate
        K.set_value(self.model.optimizer.lr, new_lr)

    def on_train_begin(self, logs={}):
        # set initial learning rate
        self.set_lr()

    def on_batch_begin(self, batch, logs={}):
        # using batch is not usefull as it starts at 0 every epoch

        # set new learning rate
        self.set_lr()

        # increment current_batch
        # increment current_batch in LrDecayCallback because order of activation
        # can differ and incorrect current_batch would be used
        self.metadata["current_batch"] += 1


class LrStepDecayCallback(Callback):
    """Set learning rate every decay_every batches according to:
       initial_learning_rate * (decay ^ (current_batch / decay every))
    """

    def __init__(self, metadata, verbose):
        super(LrStepDecayCallback, self).__init__()
        self.learning_rate = metadata["learning_rate"]
        self.decay_every = metadata["decay_every"]
        self.decay = metadata["decay"]
        self.metadata = metadata
        self.verbose = verbose

    def set_lr(self):
        # calculate learning rate
        n_decay = int(self.metadata["current_batch"] / self.decay_every)
        new_lr = self.learning_rate * (self.decay ** n_decay)

        # set new learning rate
        K.set_value(self.model.optimizer.lr, new_lr)

        # print new learning rate if verbose
        if self.verbose:
            print("\nBatch: " + str(self.metadata["current_batch"]) +
                  " New learning rate: " + str(new_lr))

    def on_train_begin(self, logs={}):
        # set initial learning rate
        self.set_lr()

    def on_batch_begin(self, batch, logs={}):
        # using batch is not usefull as it starts at 0 every epoch

        # check if learning rate has to change
        # - if we reach a new decay_every batch
        if self.metadata["current_batch"] % self.decay_every == 0:
            # change learning rate
            self.set_lr()

        # increment current_batch
        # increment current_batch in LrSchedulerCallback because order of activation
        # can differ and incorrect current_batch would be used
        self.metadata["current_batch"] += 1


class EpochDataSaverCallback(Callback):
    """Set current batch at training start
       Save metadata and Model after every epoch
    """

    def __init__(self, path, root, metadata):
        super(EpochDataSaverCallback, self).__init__()
        self.file = path
        self.root = root
        self.metadata = metadata

    def on_train_begin(self, logs={}):
        # set current_batch from metadata
        self.model.optimizer.current_batch = self.metadata['current_batch']

    def on_epoch_end(self, epoch, logs={}):
        # in case appending to logs (resuming training), get epoch number ourselves
        epoch = len(self.metadata["epoch_logs"])

        # append log to metadata
        self.metadata["epoch_logs"].append(logs)
        # save current epoch
        self.metadata["current_epoch"] = epoch

        if "val_loss" in logs:
            key = "val_loss"
        else:
            key = "loss"

        best_loss = self.metadata["epoch_logs"][self.metadata["best_epoch"]][key]
        if logs.get(key) < best_loss:
            self.metadata["best_epoch"] = epoch

        # save meta to file
        with open(self.file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        # save model to file with correct epoch
        save_file = os.path.join(self.root, FOLDER_WEIGHT,
                                 "weights.{epoch:05d}.hdf5".format(epoch=epoch))
        self.model.save(save_file)


def validate_feature_planes(verbose, dataset, model_features):
    """Verify that dataset's features match the model's expected features.
    """

    if 'features' in dataset:
        dataset_features = dataset['features'][()]
        dataset_features = dataset_features.split(",")
        if len(dataset_features) != len(model_features) or \
           any(df != mf for (df, mf) in zip(dataset_features, model_features)):
            raise ValueError("Model JSON file expects features \n\t%s\n"
                             "But dataset contains \n\t%s" % ("\n\t".join(model_features),
                                                              "\n\t".join(dataset_features)))
        elif verbose:
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
        elif verbose:
            print("Verified agreement of number of model and dataset feature planes, but cannot "
                  "verify exact match using old dataset format.")


def load_indices_from_file(shuffle_file):
    # load indices from shuffle_file
    with open(shuffle_file, "r") as f:
        indices = np.load(f)

    return indices


def save_indices_to_file(shuffle_file, indices):
    # save indices to shuffle_file
    with open(shuffle_file, "w") as f:
        np.save(f, indices)


def remove_unused_symmetries(indices, symmetries):
    # remove all rows with a symmetry not in symmetries
    remove = []

    # find all rows with incorrect symmetries
    for row in range(len(indices)):
        if not indices[row][1] in symmetries:
            remove.append(row)

    # remove rows and return new array
    return np.delete(indices, remove, 0)


def create_and_save_shuffle_indices(train_val_test, max_validation,
                                    n_total_data_size, shuffle_file_train,
                                    shuffle_file_val, shuffle_file_test):
    """ create an array with all unique state and symmetry pairs,
        calculate test/validation/training set sizes,
        seperate those sets and save them to seperate files.
    """

    symmetries = TRANSFORMATION_INDICES.values()

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

    # create training set and save to file shuffle_file_train
    train_indices = shuffle_indices[0:n_train_data]
    save_indices_to_file(shuffle_file_train, train_indices)

    # create validation set and save to file shuffle_file_val
    val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
    save_indices_to_file(shuffle_file_val, val_indices)

    # create test set and save to file shuffle_file_test
    test_indices = shuffle_indices[n_train_data + n_val_data:
                                   n_train_data + n_val_data + n_test_data]
    save_indices_to_file(shuffle_file_test, test_indices)


def load_train_val_test_indices(verbose, arg_symmetries, dataset_length, batch_size, directory):
    """Load indices from .npz files
       Remove unwanted symmerties
       Make Train set dividable by batch_size
       Return train/val/test set
    """
    # shuffle file locations for train/validation/test set
    shuffle_file_train = os.path.join(directory, FILE_TRAIN)
    shuffle_file_val = os.path.join(directory, FILE_VALIDATE)
    shuffle_file_test = os.path.join(directory, FILE_TEST)

    # load from .npz files
    train_indices = load_indices_from_file(shuffle_file_train)
    val_indices = load_indices_from_file(shuffle_file_val)
    test_indices = load_indices_from_file(shuffle_file_test)

    # used symmetries
    if arg_symmetries == "all":
        # add all symmetries
        symmetries = TRANSFORMATION_INDICES.values()
    elif arg_symmetries == "none":
        # only add standart orientation
        symmetries = [TRANSFORMATION_INDICES["noop"]]
    else:
        # add specified symmetries
        symmetries = [TRANSFORMATION_INDICES[name] for name in arg_symmetries.strip().split(",")]

    if verbose:
        print("Used symmetries: " + arg_symmetries)

    # remove symmetries not used during current run
    if len(symmetries) != len(TRANSFORMATION_INDICES):
        train_indices = remove_unused_symmetries(train_indices, symmetries)
        test_indices = remove_unused_symmetries(test_indices, symmetries)
        val_indices = remove_unused_symmetries(val_indices, symmetries)

    # Need to make sure training data is dividable by minibatch size or get
    # warning mentioning accuracy from keras
    if len(train_indices) % batch_size != 0:
        # remove first len(train_indices) % args.minibatch rows
        train_indices = np.delete(train_indices, [row for row in range(len(train_indices)
                                                  % batch_size)], 0)

    if verbose:
        print("dataset loaded")
        print("\t%d total positions" % dataset_length)
        print("\t%d total samples" % (dataset_length * len(symmetries)))
        print("\t%d total samples check" % (len(train_indices) +
              len(val_indices) + len(test_indices)))
        print("\t%d training samples" % len(train_indices))
        print("\t%d validation samples" % len(val_indices))
        print("\t%d test samples" % len(test_indices))

    return train_indices, val_indices, test_indices


def set_training_settings(resume, args, metadata, dataset_length):
    """ save all args to metadata,
        check if critical settings have been changed and prompt user about it.
        create new shuffle files if needed.
    """

    # shuffle file locations for train/validation/test set
    shuffle_file_train = os.path.join(args.out_directory, FILE_TRAIN)
    shuffle_file_val = os.path.join(args.out_directory, FILE_VALIDATE)
    shuffle_file_test = os.path.join(args.out_directory, FILE_TEST)

    # determine if new shuffle files have to be created
    save_new_shuffle_indices = not resume

    # save current symmetries to metadata
    metadata["symmetries"] = args.symmetries

    if resume:
        # check if argument model and meta model are the same
        if metadata["model_file"] != args.model:
            # verify if user really wants to use new model file
            print("the model file is different from the model file used last run: " +
                  metadata["model_file"] + ". It might be different than the old one.")
            if args.override or not confirm("Are you sure you want to use the new model?", False):  # noqa: E501
                raise ValueError("User abort after mismatch model files.")

        # check if decay_every is the same
        if args.decay_every is not None and metadata["decay_every"] != args.decay_every:
            print("Setting a new --decay-every might result in a different learning rate, restarting training might be advisable.")  # noqa: E501
            if args.override or confirm("Are you sure you want to use new decay every setting?", False):  # noqa: E501
                metadata["decay_every"] = args.decay_every

        # check if learning_rate is the same
        if args.learning_rate is not None and metadata["learning_rate"] != args.learning_rate:
            print("Setting a new --learning-rate might result in a different learning rate, restarting training might be advisable.")  # noqa: E501
            if args.override or confirm("Are you sure you want to use new learning rate setting?", False):  # noqa: E501
                metadata["learning_rate"] = args.learning_rate

        # check if decay is the same
        if args.decay is not None and metadata["decay"] != args.decay:
            print("Setting a new --decay might result in a different learning rate, restarting training might be advisable.")  # noqa: E501
            if args.override or confirm("Are you sure you want to use new decay setting?", False):  # noqa: E501
                metadata["decay"] = args.decay

        # check if batch_size is the same
        if args.minibatch is not None and metadata["batch_size"] != args.minibatch:
            print("current_batch will be recalculated, restarting training might be advisable.")
            if args.override or confirm("Are you sure you want to use new minibatch setting?", False):  # noqa: E501
                metadata["current_batch"] = int((metadata["current_batch"] *
                                                metadata["batch_size"]) / args.minibatch)
                metadata["batch_size"] = args.minibatch

        # check if max_validation is the same
        if args.max_validation is not None and metadata["max_validation"] != args.max_validation:
            print("Training set and validation set should be stricktly separated, setting a new fraction will generate a new validation set that might include data from the current traing set.")  # noqa: E501
            print("We reccommend you not to use the new max-validation setting.")
            if args.override or confirm("Are you sure you want to use new max-validation setting?", False):  # noqa: E501
                metadata["max_validation"] = args.max_validation
                # new shuffle files have to be created
                save_new_shuffle_indices = True

        # check if train_val_test is the same
        if args.train_val_test is not None and metadata["train_val_test"] != args.train_val_test:
            print("Training set and validation set should be stricktly separated, setting a new fraction will generate a new validation set that might include data from the current traing set.")  # noqa: E501
            print("We reccommend you not to use the new fraction.")
            if args.override or confirm("Are you sure you want to use new train-val-test fraction setting?", False):  # noqa: E501
                metadata["train_val_test"] = args.train_val_test
                # new shuffle files have to be created
                save_new_shuffle_indices = True

        # check if epochs is the same
        if args.epochs is not None and metadata["epochs"] != args.epochs:
            metadata["epochs"] = args.epochs

        # check if epoch_length is the same
        if args.epoch_length is not None and metadata["epoch_length"] != args.epoch_length:
            metadata["epoch_length"] = args.epoch_length

        # check if shuffle files exist and data file is the same
        if not os.path.exists(shuffle_file_train) or not os.path.exists(shuffle_file_val) or \
           not os.path.exists(shuffle_file_test) or \
           metadata["training_data"] != args.train_data or \
           metadata["available_states"] != dataset_length:
            # shuffle files do not exist or training data file has changed
            print("WARNING! shuffle files have to be recreated.")
            save_new_shuffle_indices = True
    else:
        # save all argument or default settings to metadata

        # save used model file to metadata
        metadata["model_file"] = args.model

        # save decay_every to metadata
        metadata["decay_every"] = args.decay_every

        # save learning_rate to metadata
        if args.learning_rate is not None:
            metadata["learning_rate"] = args.learning_rate
        else:
            metadata["learning_rate"] = DEFAULT_LEARNING_RATE

        # save decay to metadata
        if args.decay is not None:
            metadata["decay"] = args.decay
        else:
            metadata["decay"] = DEFAULT_DECAY

        # save batch_size to metadata
        if args.minibatch is not None:
            metadata["batch_size"] = args.minibatch
        else:
            metadata["batch_size"] = DEFAULT_BATCH_SIZE

        # save max_validation to metadata
        if args.max_validation is not None:
            metadata["max_validation"] = args.max_validation
        else:
            metadata["max_validation"] = DEFAULT_MAX_VALIDATION

        # save train_val_test to metadata
        if args.train_val_test is not None:
            metadata["train_val_test"] = args.train_val_test
        else:
            metadata["train_val_test"] = DEFAULT_TRAIN_VAL_TEST

        # save epochs to metadata
        if args.epochs is not None:
            metadata["epochs"] = args.epochs
        else:
            metadata["epochs"] = DEFAULT_EPOCH

        # save epoch_length to metadata
        if args.epoch_length is not None:
            metadata["epoch_length"] = args.epoch_length
        else:
            metadata["epoch_length"] = dataset_length

    # Record all command line args in a list so that all args are recorded even
    # when training is stopped and resumed.
    # TODO remove function argument from args... or do not save args to metadata
    # meta_args_data = metadata.get("cmd_line_args", [])
    # meta_args_data.append(vars(args))
    # metadata["cmd_line_args"] = meta_args_data

    # create and save new shuffle indices if needed
    if save_new_shuffle_indices:
        # create and save new shuffle indices to file
        create_and_save_shuffle_indices(
                metadata["train_val_test"], metadata["max_validation"], dataset_length,
                shuffle_file_train, shuffle_file_val, shuffle_file_test)

        # save total amount of states to metadata
        metadata["available_states"] = dataset_length

        # save training data file name to metadata
        metadata["training_data"] = args.train_data

        if args.verbose:
            print("created new data shuffling indices")


def train(metadata, out_directory, verbose, weight_file, meta_file):
    # set resume
    resume = weight_file is not None

    # load model from json spec
    policy = CNNPolicy.load_model(metadata["model_file"])
    model_features = policy.preprocessor.feature_list
    model = policy.model
    # load weights
    if resume:
        model.load_weights(os.path.join(out_directory, FOLDER_WEIGHT, weight_file))

    # features of training data
    dataset = h5.File(metadata["training_data"])

    # Verify that dataset's features match the model's expected features.
    validate_feature_planes(verbose, dataset, model_features)

    # create metadata file and the callback object that will write to it
    # and saves model  at the same time
    # the MetadataWriterCallback only sets 'epoch', 'best_epoch' and 'current_batch'.
    # We can add in anything else we like here
    meta_writer = EpochDataSaverCallback(meta_file, out_directory, metadata)

    # get train/validation/test indices
    train_indices, val_indices, test_indices \
        = load_train_val_test_indices(verbose, metadata['symmetries'], len(dataset["states"]),
                                      metadata["batch_size"], out_directory)

    # create dataset generators
    train_data_generator = threading_shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        train_indices,
        metadata["batch_size"],
        metadata)
    val_data_generator = threading_shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        val_indices,
        metadata["batch_size"],
        validation=True)

    # check if step decay has to be applied
    if metadata["decay_every"] is None:
        # use normal decay without momentum
        lr_scheduler_callback = LrDecayCallback(metadata)
    else:
        # use step decay
        lr_scheduler_callback = LrStepDecayCallback(metadata, verbose)

    sgd = SGD(lr=metadata["learning_rate"])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    if verbose:
        print("STARTING TRAINING")

    # check that remaining epoch > 0
    if metadata["epochs"] <= len(metadata["epoch_logs"]):
        raise ValueError("No more epochs to train!")

    model.fit_generator(
        generator=train_data_generator,
        samples_per_epoch=metadata["epoch_length"],
        nb_epoch=(metadata["epochs"] - len(metadata["epoch_logs"])),
        callbacks=[meta_writer, lr_scheduler_callback],
        validation_data=val_data_generator,
        nb_val_samples=len(val_indices))


def start_training(args):
    # set resume
    resume = args.weights is not None

    if args.verbose:
        if resume:
            print("trying to resume from %s with weights %s" %
                  (args.out_directory,
                   os.path.join(args.out_directory, FOLDER_WEIGHT, args.weights)))
        else:
            if os.path.exists(args.out_directory):
                print("directory %s exists. any previous data will be overwritten" %
                      args.out_directory)
            else:
                print("starting fresh output directory %s" % args.out_directory)

    # create all directories
    # main folder
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    # create supervised weight file folder
    weight_folder = os.path.join(args.out_directory, FOLDER_WEIGHT)
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    # metadata json file location
    meta_file = os.path.join(args.out_directory, FILE_METADATA)

    # create or load metadata from json file
    if resume and os.path.exists(meta_file):
        # load metadata
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        if args.verbose:
            print("previous metadata loaded: %d epochs. new epochs will be appended." %
                  len(metadata["epoch_logs"]))
    else:
        # create new metadata
        metadata = {
            "epoch_logs": [],
            "current_batch": 0,
            "current_epoch": 0,
            "best_epoch": 0,
            "generator_seed": None,
            "generator_sample": 0
        }

        if args.verbose:
            print("starting with empty metadata")

    # features of training data
    dataset = h5.File(args.train_data)

    # set all settings: default, from args or from metadata
    # generate new shuffle files if needed
    set_training_settings(resume, args, metadata, len(dataset["states"]))

    # start training
    train(metadata, args.out_directory, args.verbose, None, meta_file)


def resume_training(args):
    # metadata json file location
    meta_file = os.path.join(args.out_directory, FILE_METADATA)

    # load data from json file
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            metadata = json.load(f)
    else:
        raise ValueError("Metadata file not found!")

    # determine what weight file to use
    if args.weights is None:
        # newest epoch weight file from json
        weight_file = "weights.{epoch:05d}.hdf5".format(epoch=metadata["current_epoch"])
    else:
        # user weight argument
        weight_file = args.weights

    # check if training epochs has changed
    if args.epochs is not None:
        metadata["epochs"] = args.epochs

    if args.verbose:
        print("trying to resume training from %s with weights %s" %
              (meta_file, os.path.join(args.out_directory, FOLDER_WEIGHT, weight_file)))

    # start training
    train(metadata, args.out_directory, args.verbose, weight_file, meta_file)


def handle_arguments(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list
    """

    import argparse
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    # subparser is always first argument
    subparsers = parser.add_subparsers(help='sub-command help')

    # sub parser start training
    train = subparsers.add_parser('train', help='Start or resume supervised training on a policy network.')  # noqa: E501
    # required arguments
    train.add_argument("out_directory", help="directory where metadata and weights will be saved")  # noqa: E501
    train.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicy.save_model())")  # noqa: E501
    train.add_argument("train_data", help="A .h5 file of training data")
    # frequently used args
    train.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    train.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: " + str(DEFAULT_BATCH_SIZE), type=int, default=None)  # noqa: E501
    train.add_argument("--epochs", "-E", help="Total number of iterations on the data. Default: " + str(DEFAULT_EPOCH), type=int, default=None)  # noqa: E501
    train.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)  # noqa: E501
    train.add_argument("--learning-rate", "-r", help="Learning rate - how quickly the model learns at first. Default: " + str(DEFAULT_LEARNING_RATE), type=float, default=None)  # noqa: E501
    train.add_argument("--decay", "-d", help=("The rate at which learning decreases. Default: " + str(DEFAULT_DECAY)), type=float, default=None)  # noqa: E501
    train.add_argument("--decay-every", "-de", help="Use step-decay: decay --learning-rate with --decay every --decay-every batches. Default: None", type=int, default=None)  # noqa: E501
    train.add_argument("--override", help="Turn on prompt override mode", default=False, action="store_true")  # noqa: E501
    # slightly fancier args
    train.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training", default=None)  # noqa: E501
    train.add_argument("--train-val-test", help="Fraction of data to use for training/val/test. Must sum to 1. Default: " + str(DEFAULT_TRAIN_VAL_TEST), nargs=3, type=float, default=None)  # noqa: E501
    train.add_argument("--max-validation", help="maximum validation set size. default: " + str(DEFAULT_MAX_VALIDATION), type=int, default=None)  # noqa: E501
    train.add_argument("--symmetries", help="none, all or comma-separated list of transforms, subset of: noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2. Default: all", default='all')  # noqa: E501
    # function to call when start training
    train.set_defaults(func=start_training)

    # sub parser resume training
    resume = subparsers.add_parser('resume', help='Resume supervised training on a policy network. (Settings are loaded from savefile.)')  # noqa: E501
    # required arguments
    resume.add_argument("out_directory", help="directory where metadata and weight files where stored during previous session.")  # noqa: E501
    # optional argument
    resume.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    resume.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training. Default: #Newest weight file.", default=None)  # noqa: E501
    resume.add_argument("--epochs", "-E", help="Total number of iterations on the data. Defaukt: #Epochs set on previous run", type=int, default=None)  # noqa: E501
    # function to call when resume training
    resume.set_defaults(func=resume_training)

    # show help or parse arguments
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # execute function (train or resume)
    args.func(args)


if __name__ == '__main__':
    handle_arguments()
