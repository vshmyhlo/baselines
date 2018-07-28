import tensorflow as tf
from baselines.models import Model
import baselines.layers as L


# TODO: rename bn to norm
# TODO: make baseclass
# TODO: use enum for downsample type
# TODO: remove redundant `name`
# TODO: check initialization
# TODO: check regularization
# TODO: check resize-conv (upsampling)
# TODO: check training arg
# TODO: remove bias where not needed
# TODO: bn after concat
# TODO: do not use sequential in densenet for concat blocks


class ResNeXt_Bottleneck(Model):
    def __init__(self,
                 filters,
                 project,
                 kernel_initializer,
                 kernel_regularizer,
                 cardinality=32,
                 name='resnext_bottleneck'):
        assert filters % cardinality == 0
        assert project in [True, False, 'down']

        super().__init__(name=name)

        # identity
        if project == 'down':  # TODO: refactor to enum
            self.identity_conv = L.Conv2D(
                filters * 4,
                3,
                2,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)
            self.identity_bn = L.BatchNormalization()

        elif project:
            self.identity_conv = L.Conv2D(
                filters * 4,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)
            self.identity_bn = L.BatchNormalization()
        else:
            self.identity_conv = None
            self.identity_bn = None

        # conv_1
        self.conv_1 = L.Conv2D(
            filters * 2,
            1,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn_1 = L.BatchNormalization()

        # conv_2
        self.conv_2 = []
        for _ in range(cardinality):
            strides = 2 if project == 'down' else 1
            conv = L.Conv2D(
                (filters * 2) // cardinality,
                3,
                strides,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)
            self.conv_2.append(conv)

        self.bn_2 = []
        for _ in range(cardinality):
            bn = L.BatchNormalization()
            self.bn_2.append(bn)

        # conv_3
        self.conv_3 = L.Conv2D(
            filters * 4,
            1,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.bn_3 = L.BatchNormalization()

    def call(self, input, training):
        # identity
        identity = input
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        if self.identity_bn is not None:
            identity = self.identity_bn(identity, training=training)

        # conv_1
        input = self.conv_1(input)
        input = self.bn_1(input, training=training)
        input = tf.nn.relu(input)

        # conv_2
        splits = tf.split(input, len(self.conv_2), -1)
        transformations = []
        for split, conv, bn in zip(splits, self.conv_2, self.bn_2):
            split = conv(split)
            split = bn(split, training=training)
            split = tf.nn.relu(split)
            transformations.append(split)
        input = tf.concat(transformations, -1)

        # conv_3
        input = self.conv_3(input)
        input = self.bn_3(input, training=training)
        input = input + identity
        input = tf.nn.relu(input)

        return input


# TODO:
class ResNeXt_Block(Model):
    def __init__(self,
                 filters,
                 depth,
                 downsample,
                 kernel_initializer,
                 kernel_regularizer,
                 name='resnext_block'):
        super().__init__(name=name)

        layers = []

        for i in range(depth):
            if i == 0:
                project = 'down' if downsample else True
            else:
                project = False

            layer = ResNeXt_Bottleneck(
                filters, project=project, kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)
            layers.append(layer)

        self.layers = layers

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

        self.conv_1 = ResNeXt_ConvInput(
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.conv_1_max_pool = L.MaxPooling2D(3, 2, padding='same')

        self.conv_2 = ResNeXt_Block(
            filters=64, depth=3, downsample=False, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.conv_3 = ResNeXt_Block(
            filters=128, depth=4, downsample=True, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.conv_4 = ResNeXt_Block(
            filters=256, depth=6, downsample=True, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.conv_5 = ResNeXt_Block(
            filters=512, depth=3, downsample=True, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)


def main():
    image = tf.zeros((8, 224, 224, 3))

    net = ResNeXt_50()
    output = net(image, training=True)

    for k in output:
        shape = output[k].shape
        assert shape[1] == shape[2] == 224 // 2**int(k[1:]), 'invalid shape {} for layer {}'.format(shape, k)
        print(output[k])


if __name__ == '__main__':
    main()
