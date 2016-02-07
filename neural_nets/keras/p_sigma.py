from keras.models import Sequential
from keras.layers import convolutional
from keras.layers.core import Activation, Reshape
from SGD_exponential_decay import SGD_exponential_decay as SGD

### Parameters obtained from paper ###
k = 152 # depth of convolutional layers
decay = 8.664339379294006e-08 # rate of exponential learning rate decay

model = Sequential()
model.add(convolutional.Convolution2D(input_shape=(48, 19, 19), nb_filter=k, nb_row=5, nb_col=5,
                                      init='uniform', activation='relu', border_mode='same'))
for i in range(2,13):
    model.add(convolutional.Convolution2D(nb_filter=k, nb_row=3, nb_col=3,
                                          init='uniform', activation='relu', border_mode='same'))
model.add(convolutional.Convolution2D(nb_filter=1, nb_row=1, nb_col=1,
                                      init='uniform', border_mode='same'))
model.add(Reshape((19,19)))
model.add(Activation('softmax'))

sgd = SGD(lr=0.003, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
