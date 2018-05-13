import tensorflow as tf
import numpy as np

DropoutWrapper = tf.nn.rnn_cell.DropoutWrapper


class SentimentNetwork(object):

    
    def __init__(self, hdim=25, wdim=25, pdim=25, vocab_size=2000, pos_vocab_size=30, 
            num_labels=5, dropout_value=0.5, lr=0.001):

        tf.reset_default_graph()

        # placeholders
        sentences = tf.placeholder(tf.int32, [None, None], name='sentence')
        pos       = tf.placeholder(tf.int32, [None, None], name='pos')
        labels    = tf.placeholder(tf.int32, [None, ], name='label')
        mode      = tf.placeholder(tf.int32, (), name='mode')
        self.placeholders = {
                'sentence' : sentences,
                'label'    : labels,
                'mode'     : mode
                }

        # drop out
        dropout = tf.cond(
                tf.equal(mode, 0), # If
                lambda : dropout_value, # True
                lambda : 0. # False
                )

        # word embedding
        wemb = tf.get_variable(shape=[vocab_size-2, wdim], 
                dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-0.01, 0.01), 
                name='word_embedding')

        # add UNK and PAD
        wemb = tf.concat([ tf.zeros([2, wdim]), wemb ], axis=0)

        pemb = tf.get_variable(shape=[pos_vocab_size, pdim], 
                dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-0.01, 0.01), 
                name='pos_embedding')

        emb_sentence = tf.concat(
                [ tf.nn.embedding_lookup(wemb, sentences),
                tf.nn.embedding_lookup(wemb, pos) ],
                axis=-1)

        """
        # define forward and backward cells for RNN
        with tf.variable_scope('forward'):
            cell_fw = DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hdim),
                    output_keep_prob=1. - dropout)
            state_fw = cell_fw.zero_state(batch_size_, tf.float32)
        with tf.variable_scope('backward'):
            cell_bw = DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hdim),
                    output_keep_prob=1. - dropout)
            state_bw = cell_bw.zero_state(batch_size_, tf.float32)

        with tf.variable_scope('encoder') as scope:
            # encode drug sequence
            encoded_sequence, (__fsf, __fsb) = tf.nn.bidirectional_dynamic_rnn( 
                    cell_fw, cell_bw, # forward and backward cells
                    inputs= tf.nn.embedding_lookup(wemb, self.sequence), 
                    sequence_length=seqlens,
                    dtype=tf.float32)
        """

        with tf.variable_scope('rnn_cell') as scope:
            cell = DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(hdim), 
                    output_keep_prob=1. - dropout
                    )

        with tf.variable_scope('encoder') as scope:
            outputs, final_state = tf.nn.dynamic_rnn(
                    cell = cell,
                    inputs = emb_sentence,
                    sequence_length = tf.count_nonzero(sentences, axis=-1),
                    dtype=tf.float32
                    )

        logits = tf.contrib.layers.fully_connected(final_state.c, num_labels)

        self.out = {
                'prob' : tf.nn.softmax(logits),
                'pred' : tf.argmax(tf.nn.softmax(logits), axis=-1),
                'loss' : tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=labels
                        ))
                    }

        self.out['accuracy'] = tf.cast(tf.equal(
            tf.cast(self.out['pred'], tf.int32), 
            labels), tf.float32)

        self.train_op = tf.train.AdamOptimizer().minimize(self.out['loss'])


def rand_execution(netw):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(netw.out, feed_dict = {
            netw.placeholders['sentence'] : np.random.randint(0, 100, [8, 10]),
            netw.placeholders['label']    : np.random.randint(0, 4, [8, ])
            })


if __name__ == '__main__':

    netw = SentimentNetwork()
    print(rand_execution(netw))
