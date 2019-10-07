import numpy as np
import tensorflow as tf

import gp_loaddata
import gp_minibatch


'''
root = r'Datasets\Temp\Graphsage_data\toy-ppi-G.json'
graph, label, feature = gp_loaddata.load_data(root)
test_edge_itor = gp_minibatch.EdgeMinibatchIterator(graph)
'''


def test1():
    vec = tf.constant([[10, 20, 19, 30, 15]], dtype=tf.int64)
    # vec = tf.reshape(vec, [-1, 1])
    ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=vec,
        num_true=5,
        num_sampled=2,
        unique=False,
        range_max=5,
        vocab_file='',
        distortion=1.0,
        num_reserved_ids=0,
        num_shards=1,
        shard=0,
        unigrams=(0.0, 0.0, 0.0, 0.0, 0.9),
        # unigrams=(1, 2, 3, 1, 3),
    )
    # vs = ids(vec)
    with tf.Session() as sess:
        print(sess.run(ids))


if __name__ == '__main__':
    test1()
