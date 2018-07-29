import tensorflow as tf
import math
import baselines.layers as L
from baselines.models import Model, Sequential


class ClassificationSubnet(Model):
    def __init__(self,
                 num_anchors,
                 num_classes,
                 kernel_initializer,
                 kernel_regularizer,
                 name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv_pre = Sequential([
            Sequential([
                L.Conv2D(
                    256,
                    3,
                    1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                L.Normalization(),
                L.Activation(tf.nn.relu),
            ]) for _ in range(4)
        ])

        pi = 0.01
        bias_prior_initializer = tf.constant_initializer(-math.log((1 - pi) / pi))

        self.conv_out = L.Conv2D(
            num_anchors * num_classes,
            3,
            1,
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_prior_initializer)

    def call(self, input, training):
        input = self.conv_pre(input, training)
        input = self.conv_out(input)

        shape = tf.shape(input)
        input = tf.reshape(input, (shape[0], shape[1], shape[2], self.num_anchors, self.num_classes))

        return input


class RegressionSubnet(Model):
    def __init__(self,
                 num_anchors,
                 kernel_initializer,
                 kernel_regularizer,
                 name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors

        self.conv_pre = Sequential([
            Sequential([
                L.Conv2D(
                    256,
                    3,
                    1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer),
                L.Normalization(),
                L.Activation(tf.nn.relu),
            ]) for _ in range(4)
        ])

        self.conv_out = L.Conv2D(
            num_anchors * 4,
            3,
            1,
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

    def call(self, input, training):
        input = self.conv_pre(input, training)
        input = self.conv_out(input)

        shape = tf.shape(input)
        input = tf.reshape(input, (shape[0], shape[1], shape[2], self.num_anchors, 4))

        return input


class RetinaNet(Model):
    def __init__(self,
                 num_classes,
                 num_anchors,
                 # dropout_rate,  # TODO: should use dropout?
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 name='retinanet'):
        if kernel_initializer is None:
            kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        if kernel_regularizer is None:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        super().__init__(name=name)

        self.classification_subnet = ClassificationSubnet(
            num_anchors=num_anchors,
            num_classes=num_classes,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='classification_subnet')

        self.regression_subnet = RegressionSubnet(
            num_anchors=num_anchors,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name='regression_subnet')

    def call(self, input, training):
        classifications = {
            pn: self.classification_subnet(input[pn], training)
            for pn in input
        }

        regressions = {
            pn: self.regression_subnet(input[pn], training)
            for pn in input
        }

        return {
            'classifications': classifications,
            'regressions': regressions
        }
