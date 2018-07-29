from baselines.resnet import ResNet_50
from baselines.fpn import FeaturePyramidNetwork
from baselines.retinanet import RetinaNet
import tensorflow as tf


def test_output_shape():
    image = tf.zeros((8, 224, 224, 3))
    num_classes = 10
    num_anchors = 9

    backbone = ResNet_50()
    fpn = FeaturePyramidNetwork()
    retinanet = RetinaNet(num_classes=num_classes, num_anchors=num_anchors)
    output = backbone(image, training=True)
    output = fpn(output, training=True)
    output = retinanet(output, training=True)

    # TODO: do not destroy shape info which can be preserved
    for k in output['classifications']:
        assert output['classifications'][k].shape.as_list() == [None, None, None, num_anchors, num_classes]

    # TODO: do not destroy shape info which can be preserved
    for k in output['regressions']:
        assert output['regressions'][k].shape.as_list() == [None, None, None, num_anchors, 4]
