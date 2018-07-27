import tensorflow as tf
from densenet import DenseNetBC_121, DenseNetBC_169


def test_output_shape():
    image = tf.zeros((8, 224, 224, 3))

    net = DenseNetBC_121(dropout_rate=0.2)
    output = net(image, training=True)

    for k in output:
        shape = output[k].shape
        assert shape[1] == shape[2] == 224 // 2**int(k[1:]), 'invalid shape {} for layer {}'.format(shape, k)

    net = DenseNetBC_169(dropout_rate=0.2)
    output = net(image, training=True)

    for k in output:
        shape = output[k].shape
        assert shape[1] == shape[2] == 224 // 2**int(k[1:]), 'invalid shape {} for layer {}'.format(shape, k)
