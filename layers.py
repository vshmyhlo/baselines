import tensorflow as tf
import tensorflow.keras as tfk


# TODO: use_bias
class DepthwiseConv2D(tfk.layers.Layer):
    def __init__(self, kernel_size, strides, padding, use_bias, kernel_initializer, kernel_regularizer,
                 name='separable_conv2d'):
        super().__init__(name=name)

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.kernel = self.add_variable(
            'kernel',
            (self.kernel_size, self.kernel_size, input_shape[3].value, 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer)

        super().build(input_shape)

    def call(self, input):
        input = tf.nn.depthwise_conv2d(
            input,
            self.kernel,
            strides=[1, self.strides, self.strides, 1],
            padding=self.padding.upper())

        return input
