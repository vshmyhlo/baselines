import tensorflow as tf
import tensorflow.keras as tfk
from utils import Model, Sequential


class CompositeFunction(Sequential):
    def __init__(self,
                 filters,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='composite_function'):
        layers = [
            tfk.layers.BatchNormalization(),
            tfk.layers.Activation(tf.nn.relu),
            tfk.layers.Conv2D(
                filters,
                3,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tfk.layers.Dropout(dropout_rate),
        ]

        super().__init__(layers, name=name)


class BottleneckCompositeFunction(Sequential):
    def __init__(self,
                 filters,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='bottleneck_composite_function'):
        layers = [
            tfk.layers.BatchNormalization(),
            tfk.layers.Activation(tf.nn.relu),
            tfk.layers.Conv2D(
                filters * 4,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tfk.layers.Dropout(dropout_rate),
            tfk.layers.BatchNormalization(),
            tfk.layers.Activation(tf.nn.relu),
            tfk.layers.Conv2D(
                filters,
                3,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tfk.layers.Dropout(dropout_rate),
        ]

        super().__init__(layers, name=name)


class DenseNet_Block(Sequential):
    def __init__(self,
                 growth_rate,
                 depth,
                 bottleneck,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='densnet_block'):

        layers = []
        for i in range(depth):
            if bottleneck:
                layers.append(
                    BottleneckCompositeFunction(
                        growth_rate,
                        dropout_rate=dropout_rate,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        name='composite_function{}'.format(i + 1)))
            else:
                layers.append(
                    CompositeFunction(
                        growth_rate,
                        dropout_rate=dropout_rate,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        name='composite_function{}'.format(i + 1)))

        super().__init__(layers, name=name)

    def call(self, input, training):
        for layer in self.layers:
            output = layer(input, training)
            input = tf.concat([input, output], -1)

        return input


class TransitionLayer(Sequential):
    def __init__(self,
                 input_filters,
                 compression_factor,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='transition_layer'):
        self.input_filters = input_filters
        filters = int(input_filters * compression_factor)

        layers = [
            tfk.layers.BatchNormalization(),
            tfk.layers.Conv2D(
                filters,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            tfk.layers.Dropout(dropout_rate),
            tfk.layers.AveragePooling2D(2, 2, padding='same')
        ]

        super().__init__(layers, name=name)

    def call(self, input, training):
        assert input.shape[-1] == self.input_filters
        return super().call(input, training)


class DenseNetBC_ImageNet(Model):
    def __init__(self,
                 blocks,
                 growth_rate,
                 compression_factor,
                 bottleneck,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='densenet_bc_imagenet'):
        super().__init__(name=name)

        self.conv1 = Sequential([
            tfk.layers.Conv2D(
                2 * growth_rate,
                7,
                2,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='conv1'),
            tfk.layers.BatchNormalization(),
            tfk.layers.Activation(tf.nn.relu)
        ])
        self.conv1_max_pool = tfk.layers.MaxPooling2D(3, 2, padding='same')

        self.dense_block_1 = DenseNet_Block(
            growth_rate,
            depth=blocks[1],
            bottleneck=bottleneck,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='dense_block1')

        self.transition_layer_1 = TransitionLayer(
            input_filters=blocks[1] * growth_rate + 64,
            compression_factor=compression_factor,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='transition_layer_1')

        self.dense_block_2 = DenseNet_Block(
            growth_rate,
            depth=blocks[2],
            bottleneck=bottleneck,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='dense_block2')

        self.transition_layer_2 = TransitionLayer(
            input_filters=blocks[2] * growth_rate + self.transition_layer_1.layers[1].filters,  # FIXME:
            compression_factor=compression_factor,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='transition_layer_2')

        self.dense_block_3 = DenseNet_Block(
            growth_rate,
            depth=blocks[3],
            bottleneck=bottleneck,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='dense_block3')

        self.transition_layer_3 = TransitionLayer(
            input_filters=blocks[3] * growth_rate + self.transition_layer_2.layers[1].filters,  # FIXME:
            compression_factor=compression_factor,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='transition_layer_3')

        self.dense_block_4 = DenseNet_Block(
            growth_rate,
            depth=blocks[4],
            bottleneck=bottleneck,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='dense_block4')

    def call(self, input, training):
        input = self.conv1(input, training)
        C1 = input
        input = self.conv1_max_pool(input)
        input = self.dense_block_1(input, training)
        C2 = input
        input = self.transition_layer_1(input, training)
        input = self.dense_block_2(input, training)
        C3 = input
        input = self.transition_layer_2(input, training)
        input = self.dense_block_3(input, training)
        C4 = input
        input = self.transition_layer_3(input, training)
        input = self.dense_block_4(input, training)
        C5 = input

        return {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}


class DenseNetBC_121(DenseNetBC_ImageNet):
    def __init__(self,
                 dropout_rate,
                 growth_rate=32,
                 compression_factor=0.5,
                 bottleneck=True,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 name='densenet_bc_121'):
        if kernel_initializer is None:
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)

        if kernel_regularizer is None:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(
            blocks=[None, 6, 12, 24, 16],
            growth_rate=growth_rate,
            compression_factor=compression_factor,
            bottleneck=bottleneck,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name)


class DenseNetBC_169(DenseNetBC_ImageNet):
    def __init__(self,
                 dropout_rate,
                 growth_rate=32,
                 compression_factor=0.5,
                 bottleneck=True,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 name='densenet_bc_169'):
        if kernel_initializer is None:
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)

        if kernel_regularizer is None:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(
            blocks=[None, 6, 12, 32, 32],
            growth_rate=growth_rate,
            compression_factor=compression_factor,
            bottleneck=bottleneck,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name)
