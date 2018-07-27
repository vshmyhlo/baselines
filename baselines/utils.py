import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.python.util import tf_inspect


class Model(tfk.Model):
    pass


# TODO: tfk.Sequential has some bugs, so I don't use it yet
# class Sequential(tfk.Sequential):
#     pass


class Sequential(Model):
    def __init__(self, layers, name=None):
        super().__init__(name=name)

        if layers:
            for layer in layers:
                self.add(layer)

    @property
    def layers(self):
        return self._layers

    def add(self, layer):
        self._layers.append(layer)

    def call(self, input, training):
        for layer in self.layers:
            if 'training' in tf_inspect.getargspec(layer.call).args:
                input = layer(input, training=training)
            else:
                input = layer(input)

        return input
