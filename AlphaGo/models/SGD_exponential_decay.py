from keras.optimizers import SGD
from keras import backend as K
import numpy as np


class SGD_exponential_decay(SGD):
    '''Stochastic gradient descent. Same as built in SGD module, except
       the learning rate decreases as a recurrent linear function of decay,
       i.e., it doesn't depend on self.iterations.
    '''
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False,
                 *args, **kwargs):
        super(SGD_exponential_decay, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        ### THE UPDATED CALCULATION ###
        lr = self.lr * (1.0 / (1.0 + self.decay))
        self.updates = [(self.iterations, self.iterations + 1.)]
        for p, g, c in zip(params, grads, constraints):
            m = K.variable(np.zeros(K.get_value(p).shape))  # momentum
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            self.updates.append((p, c(new_p)))  # apply constraints
        return self.updates
