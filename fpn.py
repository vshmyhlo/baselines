import tensorflow as tf
import tensorflow.keras as tfk
from utils import Model, Sequential
from layers import Normalization


class UpsampleMerge(Model):
    def __init__(self,
                 kernel_initializer,
                 kernel_regularizer,
                 name='upsample_merge'):
        super().__init__(name=name)

        self.conv_lateral = Sequential([
            tfk.layers.Conv2D(
                256,
                1,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Normalization()
        ])

        self.conv_merge = Sequential([
            tfk.layers.Conv2D(
                256,
                3,
                1,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Normalization()
        ])

    # TODO: refactor arguments
    # TODO: refactor upsampling to function
    def call(self, lateral, input, training):
        lateral = self.conv_lateral(lateral, training)
        lateral_size = tf.shape(lateral)[1:3]
        input = tf.image.resize_images(
            input, lateral_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        merged = lateral + input
        merged = self.conv_merge(merged, training)

        return merged


class FeaturePyramidNetwork(Model):
    def __init__(self,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 name='feature_pyramid_network'):
        if kernel_initializer is None:
            kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        if kernel_regularizer is None:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(name=name)

        self.p6_from_c5 = Sequential([
            tfk.layers.Conv2D(
                256,
                3,
                2,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Normalization()
        ])

        self.p7_from_p6 = Sequential([
            tfk.layers.Activation(tf.nn.relu),
            tfk.layers.Conv2D(
                256,
                3,
                2,
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Normalization()
        ])

        self.p5_from_c5 = Sequential([
            tfk.layers.Conv2D(
                256,
                1,
                1,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer),
            Normalization()
        ])

        self.p4_from_c4p5 = UpsampleMerge(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='upsample_merge_c4p5')

        self.p3_from_c3p4 = UpsampleMerge(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='upsample_merge_c3p4')

    def call(self, input, training):
        P6 = self.p6_from_c5(input['C5'], training)
        P7 = self.p7_from_p6(P6, training)
        P5 = self.p5_from_c5(input['C5'], training)
        P4 = self.p4_from_c4p5(input['C4'], P5, training)
        P3 = self.p3_from_c3p4(input['C3'], P4, training)

        return {'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'P7': P7}
