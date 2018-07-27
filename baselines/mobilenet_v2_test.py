from baselines.mobilenet_v2 import MobileNetV2
import tensorflow as tf


def test_output_shape():
    image = tf.zeros((8, 224, 224, 3))

    net = MobileNetV2(dropout_rate=0.2)
    output = net(image, training=True)

    for k in output:
        shape = output[k].shape
        assert shape[1] == shape[2] == 224 // 2**int(k[1:]), 'invalid shape {} for layer {}'.format(shape, k)
