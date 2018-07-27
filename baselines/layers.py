import tensorflow as tf
import tensorflow.keras as tfk


class GroupNormalization(tfk.layers.Layer):
    def __init__(self, groups=32, eps=1e-5, name='group_normalization'):
        super().__init__(name=name)

        self.groups = groups
        self.eps = eps

    def build(self, input_shape):
        c = input_shape[-1]

        # per channel gamma and beta
        self.gamma = self.add_variable('gamma', [1, 1, 1, c], initializer=tf.constant_initializer(1.0))
        self.beta = self.add_variable('beta', [1, 1, 1, c], initializer=tf.constant_initializer(0.0))

        super().build(input_shape)

    def call(self, input, training):  # TODO: remove training
        n, h, w, _ = tf.unstack(tf.shape(input))
        _, _, _, c = input.shape

        groups = min(self.groups, c)

        # add groups
        input = tf.reshape(input, [n, h, w, groups, c // groups])

        # normalize
        mean, var = tf.nn.moments(input, [1, 2, 4], keep_dims=True)
        input = (input - mean) / tf.sqrt(var + self.eps)

        input = tf.reshape(input, [n, h, w, c]) * self.gamma + self.beta

        return input


# TODO: use_bias
class DepthwiseConv2D(tfk.layers.Layer):
    def __init__(self,
                 kernel_size,
                 strides,
                 padding,
                 use_bias,
                 kernel_initializer,
                 kernel_regularizer,
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


# preferred normalization type
Normalization = GroupNormalization
