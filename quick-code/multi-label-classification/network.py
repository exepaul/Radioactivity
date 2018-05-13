import tensorflow as tf
import numpy as np


class LogisticRegressor(object):

    def __init__(self, num_attrs, num_labels, threshold=0.5, lr=0.01):

        self.attrs = tf.placeholder(tf.float32, [None, num_attrs], name='attrs')
        self.labels= tf.placeholder(tf.int32, [None, num_labels], name='labels')

        W = tf.get_variable(shape=[num_attrs, num_labels], dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                        name='W')
        b = tf.get_variable(shape=[num_labels, ], dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                        name='b')

        logits = tf.matmul(self.attrs, W) + b

        ce = tf.nn.sigmoid_cross_entropy_with_logits( 
                labels= tf.cast(self.labels, tf.float32),
                logits=logits
                )
        loss = tf.reduce_mean(ce)

        prediction = tf.cast(tf.nn.sigmoid(logits) > threshold, tf.int32)

        accuracy = tf.cast(tf.equal(self.labels, prediction), tf.float32)

        self.out = {
                'prediction' : prediction,
                'logits'     : tf.nn.sigmoid(logits),
                'accuracy'   : tf.reduce_mean(accuracy),
                'loss'       : loss
                }

        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)

class MLP(object):

    def __init__(self, num_attrs, num_labels, hdim=100, threshold=0.5, lr=0.01):

        self.attrs = tf.placeholder(tf.float32, [None, num_attrs], name='attrs')
        self.labels= tf.placeholder(tf.int32, [None, num_labels], name='labels')

        W1 = tf.get_variable(shape=[num_attrs, hdim], dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                        name='W1')
        b1 = tf.get_variable(shape=[hdim, ], dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                        name='b1')

        W2 = tf.get_variable(shape=[hdim, num_labels], dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                        name='W2')
        b2 = tf.get_variable(shape=[num_labels, ], dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                        name='b2')


        logits = tf.matmul(tf.matmul(self.attrs, W1) + b1, W2) + b2

        ce = tf.nn.sigmoid_cross_entropy_with_logits( 
                labels= tf.cast(self.labels, tf.float32),
                logits=logits
                )
        loss = tf.reduce_mean(ce)

        prediction = tf.cast(tf.nn.sigmoid(logits) > threshold, tf.int32)

        accuracy = tf.cast(tf.equal(self.labels, prediction), tf.float32)

        self.out = {
                'prediction' : prediction,
                'logits'     : tf.nn.sigmoid(logits),
                'accuracy'   : tf.reduce_mean(accuracy),
                'loss'       : loss
                }

        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)


def random_execution(model):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(model.out, 
                feed_dict = {
                    model.attrs: np.random.uniform(-0.01, 0.01, [2, 103]),
                    model.labels : np.random.randint(0, 1, [2, 14])
                    }
                )


if __name__ == '__main__':

    model = LogisticRegressor(103, 14)
    out = random_execution(model)

    print(out)
