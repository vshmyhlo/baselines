import tensorflow as tf
import math
from normalization import Normalization
import resnet
import densenet
import mobilenet_v2
from network import Network, Sequential


# TODO: refactor with tf.layers.Layer

def build_backbone(backbone, activation, dropout_rate):
    assert backbone in ['resnet_50', 'densenet_121', 'densenet_169', 'mobilenet_v2']
    if backbone == 'resnet_50':
        return resnet.ResNeXt_50(activation=activation)
    elif backbone == 'densenet_121':
        return densenet.DenseNetBC_121(activation=activation, dropout_rate=dropout_rate)
    elif backbone == 'densenet_169':
        return densenet.DenseNetBC_169(activation=activation, dropout_rate=dropout_rate)
    elif backbone == 'mobilenet_v2':
        return mobilenet_v2.MobileNetV2(activation=activation, dropout_rate=dropout_rate)


class ClassificationSubnet(Network):
    def __init__(self,
                 num_anchors,
                 num_classes,
                 activation,
                 kernel_initializer,
                 kernel_regularizer,
                 name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.pre_conv = self.track_layer(
            Sequential([
                Sequential([
                    tf.layers.Conv2D(
                        256,
                        3,
                        1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
                    Normalization(),
                    activation,
                ]) for _ in range(4)
            ]))

        pi = 0.01
        bias_prior_initializer = tf.constant_initializer(-math.log((1 - pi) / pi))

        self.out_conv = self.track_layer(
            tf.layers.Conv2D(
                num_anchors * num_classes,
                3,
                1,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_initializer=bias_prior_initializer))

    def call(self, input, training):
        input = self.pre_conv(input, training)
        input = self.out_conv(input)

        shape = tf.shape(input)
        input = tf.reshape(input, (shape[0], shape[1], shape[2], self.num_anchors, self.num_classes))

        return input


class RegressionSubnet(Network):
    def __init__(self,
                 num_anchors,
                 activation,
                 kernel_initializer,
                 kernel_regularizer,
                 name='classification_subnet'):
        super().__init__(name=name)

        self.num_anchors = num_anchors

        self.pre_conv = self.track_layer(
            Sequential([
                Sequential([
                    tf.layers.Conv2D(
                        256,
                        3,
                        1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer),
                    Normalization(),
                    activation,
                ]) for _ in range(4)
            ]))

        self.out_conv = self.track_layer(
            tf.layers.Conv2D(
                num_anchors * 4,
                3,
                1,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer))

    def call(self, input, training):
        input = self.pre_conv(input, training)
        input = self.out_conv(input)

        shape = tf.shape(input)
        input = tf.reshape(input, (shape[0], shape[1], shape[2], self.num_anchors, 4))

        return input


class RetinaNetBase(Network):
    def __init__(self,
                 backbone,
                 levels,
                 num_classes,
                 activation,
                 dropout_rate,
                 kernel_initializer,
                 kernel_regularizer,
                 name='retinanet_base'):
        super().__init__(name=name)

        self.backbone = self.track_layer(
            build_backbone(backbone, activation=activation, dropout_rate=dropout_rate))

        if backbone == 'densenet':
            # TODO: check if this is necessary
            # DenseNet has preactivation architecture,
            # so we need to apply activation before passing features to FPN
            self.postprocess_bottom_up = {
                cn: self.track_layer(
                    Sequential([
                        Normalization(),
                        activation
                    ]))
                for cn in ['C3', 'C4', 'C5']
            }
        else:
            self.postprocess_bottom_up = None

        self.fpn = self.track_layer(
            FeaturePyramidNetwork(
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer))

        # all pyramid levels must have the same number of anchors
        num_anchors = set(levels[pn].anchor_sizes.shape[0] for pn in levels)
        assert len(num_anchors) == 1
        num_anchors = list(num_anchors)[0]

        self.classification_subnet = self.track_layer(
            ClassificationSubnet(
                num_anchors=num_anchors,  # TODO: level anchor boxes
                num_classes=num_classes,
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='classification_subnet'))

        self.regression_subnet = self.track_layer(
            RegressionSubnet(
                num_anchors=num_anchors,  # TODO: level anchor boxes
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='regression_subnet'))

    def call(self, input, training):
        bottom_up = self.backbone(input, training)

        if self.postprocess_bottom_up is not None:
            bottom_up = {
                cn: self.postprocess_bottom_up[cn](bottom_up[cn], training)
                for cn in ['C3', 'C4', 'C5']
            }

        top_down = self.fpn(bottom_up, training)

        classifications = {
            k: self.classification_subnet(top_down[k], training)
            for k in top_down
        }

        regressions = {
            k: self.regression_subnet(top_down[k], training)
            for k in top_down
        }

        return {
            'classifications': classifications,
            'regressions': regressions
        }


class RetinaNet(Network):
    def __init__(self, backbone, levels, num_classes, activation, dropout_rate, name='retinanet'):
        super().__init__(name=name)

        self.levels = levels

        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)

        self.base = RetinaNetBase(
            backbone=backbone,
            levels=levels,
            num_classes=num_classes,
            activation=activation,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

    def call(self, input, training):
        return self.base(input, training)
