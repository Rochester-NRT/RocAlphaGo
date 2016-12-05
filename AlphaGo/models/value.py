from keras.models import Sequential
from keras.layers import convolutional
from keras.layers.core import Dense, Flatten
# from SGD_exponential_decay import SGD_exponential_decay as SGD

# Parameters obtained from paper
K = 152                        # depth of convolutional layers
LEARNING_RATE = .003           # initial learning rate
DECAY = 8.664339379294006e-08  # rate of exponential learning_rate decay


class value_trainer:
    def __init__(self):
        self.model = Sequential()
        self.model.add(convolutional.Convolution2D(
            input_shape=(49, 19, 19), nb_filter=K, nb_row=5, nb_col=5,
            init='uniform', activation='relu', border_mode='same'))
        for i in range(2, 13):
            self.model.add(convolutional.Convolution2D(
                nb_filter=K, nb_row=3, nb_col=3,
                init='uniform', activation='relu', border_mode='same'))

        self.model.add(convolutional.Convolution2D(
            nb_filter=1, nb_row=1, nb_col=1,
            init='uniform', activation='linear', border_mode='same'))
        self.model.add(Flatten())
        self.model.add(Dense(256, init='uniform'))
        self.model.add(Dense(1, init='uniform', activation="tanh"))

        # sgd = SGD(lr=LEARNING_RATE, decay=DECAY)
        # self.model.compile(loss='mean_squared_error', optimizer=sgd)

    def get_samples(self):
        # TODO non-terminating loop that draws training samples uniformly at random
        pass

    def train(self):
        # TODO use self.model.fit_generator to train from data source
        pass


if __name__ == '__main__':
    trainer = value_trainer()
    # TODO command line instantiation
