import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, n_layer, activation_func=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weight'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, name='b')
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('comp_act'):
            net_compute = tf.matmul(inputs, Weights)+biases
            if activation_func is None:
                net_compute = net_compute
            else:
                net_compute = activation_func(net_compute)
            tf.summary.histogram(layer_name+'/net_compute', net_compute)
            return net_compute


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

layer_1 = add_layer(xs, 1, 30, n_layer=1, activation_func=tf.nn.relu)
prediction = add_layer(layer_1, 30, 1, n_layer=2, activation_func=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                        reduction_indices=[1]), reduction_indices=[0])
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir=r"D:\myfile\data\Tensorflow\logs", graph=sess.graph)
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for tra_a in range(2000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if tra_a % 50 == 0:
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, tra_a)
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)

# tensorboard --logdir D:\myfile\data\Tensorflow\logs --port 6006
