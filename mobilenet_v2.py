import tensorflow as tf
import tensorflow.keras as tfk
import layers


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
# TODO: move everything to `build`


class Bottleneck(tfk.Model):
    def __init__(self, filters, strides, expansion_factor, dropout_rate, kernel_initializer,
                 kernel_regularizer,
                 name='bottleneck'):
        super().__init__(name=name)

        self.filters = filters
        self.strides = strides
        self.expansion_factor = expansion_factor
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.expand_conv = tfk.Sequential([
            tfk.layers.Conv2D(
                input_shape[3].value * self.expansion_factor,
                1,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer),
            tfk.layers.BatchNormalization(),
            tfk.layers.Activation(tf.nn.relu6),
            tfk.layers.Dropout(self.dropout_rate)
        ])

        self.depthwise_conv = tfk.Sequential([
            layers.DepthwiseConv2D(
                3, strides=self.strides, padding='same', use_bias=False, kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer),
            tfk.layers.BatchNormalization(),
            tfk.layers.Activation(tf.nn.relu6),
            tfk.layers.Dropout(self.dropout_rate)
        ])

        self.linear_conv = tfk.Sequential([
            tfk.layers.Conv2D(
                self.filters, 1, use_bias=False, kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer),
            tfk.layers.BatchNormalization(),
            tfk.layers.Dropout(self.dropout_rate)
        ])

        super().build(input_shape)

    def call(self, input, training):
        identity = input

        input = self.expand_conv(input, training)
        input = self.depthwise_conv(input, training)
        input = self.linear_conv(input, training)

        if input.shape == identity.shape:
            input = input + identity

        return input


class MobileNetV2(tfk.Model):
    def __init__(self, dropout_rate, name='mobilenet_v2'):
        super().__init__(name=name)

        self.dropout_rate = dropout_rate
        self.kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=4e-5)

    def build(self, input_shape):
        self.input_conv = tfk.Sequential([
            tfk.layers.Conv2D(
                32, 3, strides=2, padding='same', use_bias=False, kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer),
            tfk.layers.BatchNormalization(),
            tfk.layers.Activation(tf.nn.relu6),
            tfk.layers.Dropout(self.dropout_rate)
        ])

        self.bottleneck_1_1 = Bottleneck(
            16, expansion_factor=1, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)

        self.bottleneck_2_1 = Bottleneck(
            24, expansion_factor=6, strides=2, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_2_2 = Bottleneck(
            24, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)

        self.bottleneck_3_1 = Bottleneck(
            32, expansion_factor=6, strides=2, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_3_2 = Bottleneck(
            32, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_3_3 = Bottleneck(
            32, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)

        self.bottleneck_4_1 = Bottleneck(
            64, expansion_factor=6, strides=2, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_4_2 = Bottleneck(
            64, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_4_3 = Bottleneck(
            64, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_4_4 = Bottleneck(
            64, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)

        self.bottleneck_5_1 = Bottleneck(
            96, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_5_2 = Bottleneck(
            96, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_5_3 = Bottleneck(
            96, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)

        self.bottleneck_6_1 = Bottleneck(
            160, expansion_factor=6, strides=2, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_6_2 = Bottleneck(
            160, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bottleneck_6_3 = Bottleneck(
            160, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)

        self.bottleneck_7_1 = Bottleneck(
            320, expansion_factor=6, strides=1, dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)

        self.output_conv = tfk.Sequential([
            tfk.layers.Conv2D(
                32, 1, use_bias=False, kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer),
            tfk.layers.BatchNormalization(),
            tfk.layers.Activation(tf.nn.relu6),
            tfk.layers.Dropout(self.dropout_rate)
        ])

        super().build(input_shape)

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


def main():
    image = tf.zeros((8, 224, 224, 3))

    net = MobileNetV2(dropout_rate=0.2)
    output = net(image, training=True)

    for k in output:
        shape = output[k].shape
        assert shape[1] == shape[2] == 224 // 2**int(k[1:]), 'invalid shape {} for layer {}'.format(shape, k)
        print(output[k])


if __name__ == '__main__':
    main()
