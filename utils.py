import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.python.util import tf_inspect


class Model(tfk.Model):
    pass


# TODO: tfk.Sequential has some bugs, so I don't use it yet
# class Sequential(tfk.Sequential):
#     pass


class Sequential(Model):
    def __init__(self, layers, name='sequential'):
        super().__init__(name=name)
        self.sequential_layers = layers

    def call(self, input, training):
        for l in self.sequential_layers:
            if 'training' in tf_inspect.getargspec(l.call).args:
                input = l(input, training=training)
            else:
                input = l(input)

        return input
