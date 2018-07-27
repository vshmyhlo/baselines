import tensorflow as tf
import tensorflow.keras as tfk
from baselines.utils import Model, Sequential


# TODO: make baseclass
# TODO: use maxpool as 2nd layer
# TODO: fused bn

class ConvInput(Sequential):
    def __init__(self,
                 filters,
                 kernel_initializer,
                 kernel_regularizer,
                 name='conv_input'):
        layers = [
            tfk.layers.Conv2D(
                filters,
                7,
                strides=2,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tfk.layers.BatchNormalization(fused=True),
            tfk.layers.Activation(tf.nn.relu)
        ]

        super().__init__(layers, name=name)


class Bottleneck(Model):
    def __init__(self,
                 filters,
                 kernel_initializer,
                 kernel_regularizer,
                 name='bottleneck'):
        super().__init__(name=name)

        self.conv_1 = tfk.layers.Conv2D(
            filters // 4,
            1,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn_1 = tfk.layers.BatchNormalization(fused=True)

        self.conv_2 = tfk.layers.Conv2D(
            filters // 4,
            3,
            padding='same',
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn_2 = tfk.layers.BatchNormalization(fused=True)

        self.conv_3 = tfk.layers.Conv2D(
            filters,
            1,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn_3 = tfk.layers.BatchNormalization(fused=True)

    def call(self, input, training):
        identity = input

        input = self.conv_1(input)
        input = self.bn_1(input, training=training)
        input = tf.nn.relu(input)

        input = self.conv_2(input)
        input = self.bn_2(input, training=training)
        input = tf.nn.relu(input)

        input = self.conv_3(input)
        input = self.bn_3(input, training=training)
        input = input + identity
        input = tf.nn.relu(input)

        return input


class BottleneckDown(Model):
    def __init__(self,
                 filters,
                 kernel_initializer,
                 kernel_regularizer,
                 name='bottleneck_down'):
        super().__init__(name=name)

        self.conv_identity = tfk.layers.Conv2D(
            filters,
            3,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn_identity = tfk.layers.BatchNormalization(fused=True)

        self.conv_1 = tfk.layers.Conv2D(
            filters // 4,
            1,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn_1 = tfk.layers.BatchNormalization(fused=True)

        self.conv_2 = tfk.layers.Conv2D(
            filters // 4,
            3,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn_2 = tfk.layers.BatchNormalization(fused=True)

        self.conv_3 = tfk.layers.Conv2D(
            filters,
            1,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn_3 = tfk.layers.BatchNormalization(fused=True)

    def call(self, input, training):
        identity = input
        identity = self.conv_identity(identity)
        identity = self.bn_identity(identity, training=training)

        input = self.conv_1(input)
        input = self.bn_1(input, training=training)
        input = tf.nn.relu(input)

        input = self.conv_2(input)
        input = self.bn_2(input, training=training)
        input = tf.nn.relu(input)

        input = self.conv_3(input)
        input = self.bn_3(input, training=training)
        input = input + identity
        input = tf.nn.relu(input)

        return input


class ResBlock(Sequential):
    def __init__(self,
                 filters,
                 depth,
                 kernel_initializer,
                 kernel_regularizer,
                 name='res_block'):
        assert filters % 4 == 0
        assert filters // 4 >= 64

        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(BottleneckDown(
                    filters, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
            else:
                layers.append(Bottleneck(
                    filters, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))

        super().__init__(layers, name=name)


class ResNet(Model):
    def call(self, input, training):
        input = self.conv_input(input, training=training)
        C1 = input
        input = self.res_block_1(input, training=training)
        C2 = input
        input = self.res_block_2(input, training=training)
        C3 = input
        input = self.res_block_3(input, training=training)
        C4 = input
        input = self.res_block_4(input, training=training)
        C5 = input

        return {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}


class ResNet_50(ResNet):
    def __init__(self,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 name='resnet_50'):
        if kernel_initializer is None:
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)

        if kernel_regularizer is None:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(name=name)

        self.conv_input = ConvInput(
            64, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.res_block_1 = ResBlock(
            256, depth=3, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.res_block_2 = ResBlock(
            512, depth=4, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.res_block_3 = ResBlock(
            1024, depth=6, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.res_block_4 = ResBlock(
            2048, depth=3, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
