import tensorflow as tf
a = tf.constant([2, 3])
b = tf.constant([4, 5])
c = tf.concat([a, b], axis=-1)
print(c)
