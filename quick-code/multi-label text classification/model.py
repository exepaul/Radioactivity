import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


batch_size=50

# iteration=int(len(dataset)//batch_size)

epoch=40



class movies_classifier():

    def __init__(self):

        tf.reset_default_graph()


        input_sentences= tf.placeholder(name='input',shape=[None,None],dtype=tf.int32)

        labels = tf.placeholder(name='labels',shape=[None,None],dtype=tf.int32)

        mode = tf.placeholder(tf.int32, (), name='mode')

        self.placeholder={'input':input_sentences,'output':labels,'mode':mode}

        dropout = tf.cond(
            tf.equal(mode, 0),  # If
            lambda: 0.5,  # True
            lambda: 0.  # False
        )


        sequence_leng=tf.count_nonzero(input_sentences,axis=-1)  #10x10x50

        word_embedding=tf.get_variable(name='wemb',shape=[36500,50],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

        word_lookup=tf.nn.embedding_lookup(word_embedding,input_sentences)

        with tf.variable_scope('encoder'):

            fr_cell=rnn.LSTMCell(num_units=250)

            dropout_fr=rnn.DropoutWrapper(cell=fr_cell,output_keep_prob=1.-dropout)

        with tf.variable_scope('encoder'):

            bw_cell=rnn.LSTMCell(num_units=250)

            dropout_bw=rnn.DropoutWrapper(cell=bw_cell,output_keep_prob=1.-dropout)

        with tf.variable_scope('encoder') as scope:

            model,(fs,fh)=tf.nn.bidirectional_dynamic_rnn(dropout_fr,dropout_bw,inputs=word_lookup,sequence_length=sequence_leng,dtype=tf.float32)



        fs_transpose=tf.transpose(model[0],[1,0,2])

        bw_transpose=tf.transpose(model[1],[1,0,2])

        concat_result=tf.concat([fs_transpose[-1],bw_transpose[-1]],axis=-1)



        #






        #
        weights=tf.get_variable(name='weights',shape=[2*250,28],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))
        # #
        bias=tf.get_variable(name='bias',shape=[28,],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

        logits=tf.matmul(concat_result,weights) + bias

        ce=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,tf.float32),logits=logits)

        loss=tf.reduce_mean(ce)
        # #

        pred=tf.nn.sigmoid(logits)

        threshold=0.5



        #
        last_prediction=tf.cast(pred>threshold,tf.int32)
        #
        accuracy=tf.cast(tf.equal(labels,last_prediction),tf.float32)
        #
        #
        #
        self.out={'prediction':last_prediction,'accuracy':tf.reduce_mean(accuracy),'loss':loss,'logits':logits}
        #
        self.train_op=tf.train.AdamOptimizer().minimize(loss)





















