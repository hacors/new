import tensorflow as tf

if __name__ == '__main__':

    n = 1

    X = tf.placeholder(tf.float32, shape=[None, n], name='input')  # input
    W1 = tf.get_variable("W1", [n, 1], initializer=tf.ones_initializer())
    b1 = tf.constant(0, tf.float32)
    Z = tf.matmul(X, W1) + b1     # outputs

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # sess.run(Z,feed_dict={X:})

        # 保存模型
        tf.saved_model.simple_save(sess, r'Competition\Mathe_competition\Signal\model\simdemo\model', inputs={"myInput": X}, outputs={"myOutput": Z})
