import tensorflow as tf
import baselines.layers as L
from baselines.models import Model, Sequential


# TODO: do not use _ prefix anywhere
# TODO: refactor without network
# TODO: remove `track_layer(...)` stuff
# TODO: initialization
# TODO: reegularization
# TODO: batchnorm
# TODO: activation parameter
# TODO: check relu6
# TODO: dropout
# TODO: private fields
# TODO: check shapes


class Bottleneck(Model):
    def __init__(self,
                 filters,
                 strides,
                 expansion_factor,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='bottleneck'):
        super().__init__(name=name)

        self.expand_conv = Sequential([
            L.Conv2D(
                filters * expansion_factor,  # FIXME: should be `input_shape[3].value * expansion_factor`
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            L.BatchNormalization(),
            L.Activation(tf.nn.relu6),
            L.Dropout(dropout_rate)
        ])

        self.depthwise_conv = Sequential([
            L.DepthwiseConv2D(
                3,
                strides=strides,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            L.BatchNormalization(),
            L.Activation(tf.nn.relu6),
            L.Dropout(dropout_rate)
        ])

        self.linear_conv = Sequential([
            L.Conv2D(
                filters,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            L.BatchNormalization(),
            L.Dropout(dropout_rate)
        ])

    def call(self, input, training):
        identity = input

        input = self.expand_conv(input, training)
        input = self.depthwise_conv(input, training)
        input = self.linear_conv(input, training)

        if input.shape == identity.shape:
            print('same shape')
            input = input + identity

        return input


class MobileNetV2(Model):
    def __init__(self, dropout_rate, name='mobilenet_v2'):
        super().__init__(name=name)

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=4e-5)

        self.input_conv = Sequential([
            L.Conv2D(
                32,
                3,
                strides=2,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            L.BatchNormalization(),
            L.Activation(tf.nn.relu6),
            L.Dropout(dropout_rate)
        ])

        self.bottleneck_1_1 = Bottleneck(
            16, expansion_factor=1, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

        self.bottleneck_2_1 = Bottleneck(
            24, expansion_factor=6, strides=2, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_2_2 = Bottleneck(
            24, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

        self.bottleneck_3_1 = Bottleneck(
            32, expansion_factor=6, strides=2, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_3_2 = Bottleneck(
            32, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_3_3 = Bottleneck(
            32, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

        self.bottleneck_4_1 = Bottleneck(
            64, expansion_factor=6, strides=2, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_4_2 = Bottleneck(
            64, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_4_3 = Bottleneck(
            64, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_4_4 = Bottleneck(
            64, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

        self.bottleneck_5_1 = Bottleneck(
            96, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_5_2 = Bottleneck(
            96, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_5_3 = Bottleneck(
            96, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

        self.bottleneck_6_1 = Bottleneck(
            160, expansion_factor=6, strides=2, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_6_2 = Bottleneck(
            160, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bottleneck_6_3 = Bottleneck(
            160, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

        self.bottleneck_7_1 = Bottleneck(
            320, expansion_factor=6, strides=1, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

        self.output_conv = Sequential([
            L.Conv2D(
                32,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            L.BatchNormalization(),
            L.Activation(tf.nn.relu6),
            L.Dropout(dropout_rate)
        ])

    def call(self, input, training):
        input = self.input_conv(input, training)

        input = self.bottleneck_1_1(input, training)
        C1 = input

        input = self.bottleneck_2_1(input, training)
        input = self.bottleneck_2_2(input, training)
        C2 = input

        input = self.bottleneck_3_1(input, training)
        input = self.bottleneck_3_2(input, training)
        input = self.bottleneck_3_3(input, training)
        C3 = input

        input = self.bottleneck_4_1(input, training)
        input = self.bottleneck_4_2(input, training)
        input = self.bottleneck_4_3(input, training)
        input = self.bottleneck_4_4(input, training)

        input = self.bottleneck_5_1(input, training)
        input = self.bottleneck_5_2(input, training)
        input = self.bottleneck_5_3(input, training)
        C4 = input

        input = self.bottleneck_6_1(input, training)
        input = self.bottleneck_6_2(input, training)
        input = self.bottleneck_6_3(input, training)

        input = self.bottleneck_7_1(input, training)

        input = self.output_conv(input, training)
        C5 = input

        return {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}
