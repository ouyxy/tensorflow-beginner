# coding: utf8

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]  # 添加一个轴，有原来的(6000,28,28) => (6000,28,28,1)
x_test = x_test[..., tf.newaxis]

print()
