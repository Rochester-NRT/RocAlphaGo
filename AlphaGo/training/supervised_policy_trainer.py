import os, argparse
import cPickle as pickle
import random
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from AlphaGo.models.policy import CNNPolicy

class supervised_policy_trainer:
    def __init__(self,train_batch_size,test_batch_size=None,
                 learning_rate=.003,decay=.0001,nb_epoch=10):
    	"""Construct a supervised-learning policy trainer.

    	Training parameters:
    	- train_batch_size:  Number of training samples per SGD minibatch (no default)
        - test_batch_size:   Number of test samples to use when estimating model accuracy.
                             If None, whole folder is used. (default None)
    	- learning_rate:     Initial learning rate for SGD (default .003)
    	- decay:             Rate of learning rate decay (default .0001)
    	- nb_epoch:          Number of iterations through training set (default 10)
        """
        self.learning_rate = learning_rate
        self.decay =  decay
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.nb_epoch = nb_epoch

        # These are 8 symmetric groups used to randomly transform training samples,
            # which mitigates overfitting.
        self.BOARD_TRANSFORMATIONS = [
            lambda feature: feature,
            lambda feature: np.rot90(feature,1),
            lambda feature: np.rot90(feature,2),
            lambda feature: np.rot90(feature,3),
            lambda feature: np.fliplr(feature),
            lambda feature: np.flipud(feature),
            lambda feature: np.transpose(feature),
            lambda feature: np.fliplr(np.rot90(feature,1))
        ]

    def train(self,model,train_folder,test_folder,model_folder=None,checkpt_prefix="weights"):
        '''Fit an arbitrary keras model according to the training parameters.

        Options:
        - model:          A keras model to fit. Assumed not to be compiled.
    	- train_folder:   Folder of samples for training.
        - test_folder:    Folder of samples for accuracy validation.
        - model_folder:   Folder to save the model as a .hdf5 file at the end of every epoch.
                          If omitted, no models are saved.
        - checkpt_prefix: String to prepend to each saved model file.
                          Used for definiteness across training sessions.
        '''
        # 1. Compile model
        sgd = SGD(lr=self.learning_rate,decay=self.decay)
        model.compile(loss='binary_crossentropy', optimizer=sgd)

        # 2. Construct generators to fetch train and test data
        X_shape = model.get_config()['layers'][0]['input_shape']
        y_shape = X_shape[-2:] # class labels will always be board x board

        # generator setup returns a tuple of dataset size and generator for retrieving samples
        trainset_size, train_generator = self._setup_generator(train_folder,X_shape,y_shape,
                                            self.train_batch_size,sym_transform=True)
        testset_size, test_generator = self._setup_generator(test_folder,X_shape,y_shape,self.test_batch_size)

        self.train_batch_size = trainset_size if self.train_batch_size is None else self.train_batch_size
        self.test_batch_size = testset_size if self.test_batch_size is None else self.test_batch_size

        # 3. Train. Save model to new file each epoch.
        print "Training prepared successfully. Commencing training on", str(trainset_size), \
            "training samples in batches of", str(self.train_batch_size), "."
        print "Testing will occur on", str(self.test_batch_size), \
            "samples drawn without replacement from a pool of", str(testset_size), "."
        if model_folder is None:
            model.fit_generator(generator=train_generator,samples_per_epoch=self.train_batch_size,nb_epoch=self.nb_epoch,
                                validation_data=test_generator,nb_val_samples=self.test_batch_size,show_accuracy=True)
        else:
            model_path = os.path.join(model_folder,checkpt_prefix + ".{epoch:02d}-{val_loss:.2f}.hdf5")
            checkpointer = ModelCheckpoint(filepath=model_path)
            model.fit_generator(generator=train_generator,samples_per_epoch=self.train_batch_size,nb_epoch=self.nb_epoch,
                                validation_data=test_generator,nb_val_samples=self.test_batch_size,show_accuracy=True,callbacks=[checkpointer])

    def _setup_generator(self,folder,X_shape,y_shape,num_samples,sym_transform=False):
        # Returns number of samples in folder and a generator yielding batches of them
        filenames = [filename for filename in os.listdir(folder) if filename[-4:] == '.pkl']
        if num_samples is None: num_samples = len(filenames)
        def generator():
            while True:
                sample_filenames = random.sample(filenames,num_samples)
                X = np.empty((num_samples,) + X_shape, dtype='float64')
                y = np.empty((num_samples,) + y_shape, dtype='float64')
                for index,filename in enumerate(sample_filenames):
                    feature_input, label = self._prep_sample(folder,filename,sym_transform)
                    X[index] = feature_input
                    y[index] = label
                yield (X,y)
        return(len(filenames),generator())

    def _prep_sample(self,folder,filename,sym_transform):
        with open(os.path.join(folder,filename),'r') as sample_filename:
            feature_input, label = pickle.load(sample_filename)
        if sym_transform: # randomly transform sample to some symmetric version of itself
            transform = random.choice(self.BOARD_TRANSFORMATIONS)
            # apply tranform at every depth
            feature_input[0] = np.array([transform(feature) for feature in feature_input[0]])
            label[0] = transform(label[0])
        return feature_input[0],label[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    parser.add_argument("train_folder", help="Path to folder of training samples")
    parser.add_argument("test_folder", help="Path to folder of testing samples")
    parser.add_argument("input_dim",help="Feature depth of input tensors.",type=int)
    parser.add_argument("train_batch_size",help="Number of samples per SGD batch.",type=int)
    parser.add_argument("-test_batch_size",help="Number of samples per SGD batch. If omitted, all samples in folder will be used.",type=int,default=None)
    parser.add_argument("-model_folder", help="Path to folder where the model params will be saved after each epoch. Default: None",default=None)
    parser.add_argument("-nb_epoch",help="Total number of iterations on the data. Default: 10",type=int,default=10)
    parser.add_argument("-learning_rate",help="How quickly the model learns at first. A (small) number between 0 and 1. Default: .03",type=float,default=.03)
    parser.add_argument("-decay",help="The rate at which learning decreases. Default: .0001",type=float,default=.0001)
    args = parser.parse_args()

    net = CNNPolicy.create_network(input_dim=args.input_dim)

    trainer = supervised_policy_trainer(train_batch_size=args.train_batch_size,test_batch_size=args.test_batch_size,
                                        learning_rate=args.learning_rate,decay=args.decay,nb_epoch=args.nb_epoch)
    trainer.train(net,args.train_folder,args.test_folder,args.model_folder)
