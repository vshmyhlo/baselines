import tensorflow as tf
from baselines.models import Model, Sequential
import baselines.layers as L
import enum

# TODO: remove redundant `name`
# TODO: check initialization
# TODO: check regularization
# TODO: check training arg
# TODO: remove bias where not needed
# TODO: norm after concat

ProjectionType = enum.Enum('ProjectionType', ['NONE', 'DOWN', 'CONV'])


class ResNeXt_Bottleneck(Model):
    def __init__(self,
                 filters,
                 projection_type,
                 kernel_initializer,
                 kernel_regularizer,
                 cardinality=32,
                 name='resnext_bottleneck'):
        assert filters % cardinality == 0
        assert projection_type in ProjectionType

        super().__init__(name=name)

        # identity
        if projection_type is ProjectionType.DOWN:
            self.conv_identity = Sequential([
                L.Conv2D(
                    filters * 4,
                    3,
                    2,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                L.BatchNormalization()
            ])
        elif projection_type is ProjectionType.CONV:
            self.conv_identity = Sequential([
                L.Conv2D(
                    filters * 4,
                    1,
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                L.BatchNormalization()
            ])
        elif projection_type is ProjectionType.NONE:
            self.conv_identity = None

        # conv_1
        self.conv_1 = Sequential([
            L.Conv2D(
                filters * 2,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            L.BatchNormalization(),
            L.Activation(tf.nn.relu)
        ])

        # conv_2
        self.conv_2 = []
        for _ in range(cardinality):
            strides = 2 if projection_type is ProjectionType.DOWN else 1

            self.conv_2.append(
                L.Conv2D(
                    (filters * 2) // cardinality,
                    3,
                    strides,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))

        self.bn_2 = L.BatchNormalization()

        # conv_3
        self.conv_3 = Sequential([
            L.Conv2D(
                filters * 4,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            L.BatchNormalization()
        ])

    def call(self, input, training):
        # identity
        identity = input
        if self.conv_identity is not None:
            identity = self.conv_identity(identity, training=training)

        # conv_1
        input = self.conv_1(input, training=training)

        # conv_2
        splits = tf.split(input, len(self.conv_2), -1)
        transformations = []
        for split, conv in zip(splits, self.conv_2):
            split = conv(split)
            transformations.append(split)
        input = tf.concat(transformations, -1)
        input = self.bn_2(input, training=training)
        input = tf.nn.relu(input)

        # conv_3
        input = self.conv_3(input, training=training)
        input = input + identity
        input = tf.nn.relu(input)

        return input


# TODO: better names
class ResNeXt_Block(Model):
    def __init__(self,
                 filters,
                 depth,
                 downsample,
                 kernel_initializer,
                 kernel_regularizer,
                 name='resnext_block'):
        super().__init__(name=name)

        self.layers = []

        for i in range(depth):
            if i == 0:
                projection_type = ProjectionType.DOWN if downsample else ProjectionType.CONV
            else:
                projection_type = ProjectionType.NONE

            self.layers.append(
                ResNeXt_Bottleneck(
                    filters,
                    projection_type=projection_type,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))

    def call(self, input, training):
        for layer in self.layers:
            input = layer(input, training=training)

        return input


class ResNeXt_ConvInput(Model):
    def __init__(self,
                 kernel_initializer,
                 kernel_regularizer,
                 name='resnext_conv1'):
        super().__init__(name=name)

        self.conv = L.Conv2D(
            64,
            7,
            2,
            padding='same',
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn = L.BatchNormalization()

    def call(self, input, training):
        input = self.conv(input)
        input = self.bn(input, training=training)
        input = tf.nn.relu(input)

        return input


class ResNeXt(Model):
    def call(self, input, training):
        input = self.conv_1(input, training=training)
        C1 = input
        input = self.conv_1_max_pool(input)
        input = self.conv_2(input, training=training)
        C2 = input
        input = self.conv_3(input, training=training)
        C3 = input
        input = self.conv_4(input, training=training)
        C4 = input
        input = self.conv_5(input, training=training)
        C5 = input

        return {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}


class ResNeXt_50(ResNeXt):
    def __init__(self,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 name='resnext_v2_50'):
        if kernel_initializer is None:
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)

        if kernel_regularizer is None:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(name=name)

        self.conv_1 = ResNeXt_ConvInput(kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.conv_1_max_pool = L.MaxPooling2D(3, 2, padding='same')

        self.conv_2 = ResNeXt_Block(
            filters=64,
            depth=3,
            downsample=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.conv_3 = ResNeXt_Block(
            filters=128,
            depth=4,
            downsample=True,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.conv_4 = ResNeXt_Block(
            filters=256,
            depth=6,
            downsample=True,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.conv_5 = ResNeXt_Block(
            filters=512,
            depth=3,
            downsample=True,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
