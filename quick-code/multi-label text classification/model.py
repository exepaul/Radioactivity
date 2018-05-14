import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn



class movies_classifier():

    def __init__(self):

        input_sentences= tf.placeholder(name='input',shape=[None,None],dtype=tf.int32)

        labels = tf.placeholder(name='labels',shape=[None,None],dtype=tf.int32)

        self.placeholder={'input':input_sentences,'output':labels}


        sequence_leng=tf.count_nonzero(input_sentences,axis=-1)  #10x10x50

        word_embedding=tf.get_variable(name='wemb',shape=[36500,50],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))

        word_lookup=tf.nn.embedding_lookup(word_embedding,input_sentences)

        with tf.variable_scope('encoder'):

            fr_cell=rnn.LSTMCell(num_units=250)

            dropout_fr=rnn.DropoutWrapper(cell=fr_cell,output_keep_prob=0.5)

        with tf.variable_scope('encoder'):

            bw_cell=rnn.LSTMCell(num_units=250)

            dropout_bw=rnn.DropoutWrapper(cell=bw_cell,output_keep_prob=0.5)

        with tf.variable_scope('encoder') as scope:

            model,(fs,fh)=tf.nn.bidirectional_dynamic_rnn(dropout_fr,dropout_bw,inputs=word_lookup,sequence_length=sequence_leng,dtype=tf.float32)



        fs_transpose=tf.transpose(model[0],[1,0,2])

        bw_transpose=tf.transpose(model[1],[1,0,2])

        concat_result=tf.concat([fs_transpose[-1],bw_transpose[-1]],axis=-1)







        # self.out={'out':model,'each':fs.c,'result':concat_result,'em':  word_lookup}

        # transform_output=tf.transpose(model,[1,0,2])
        #
        weights=tf.get_variable(name='weights',shape=[2*250,28],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))
        # #
        bias=tf.get_variable(name='bias',shape=[28,],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))
        # #
        # #
        logits=tf.matmul(concat_result,weights) + bias
        #
        # #
        ce=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,tf.float32),logits=logits)
        #
        #
        # self.out={'ce':ce,'logits':logits,'result':concat_result}

        loss=tf.reduce_mean(ce)
        # #
        threshold=0.5

        # logits = tf.matmul(self.attrs, W) + b
        #
        # ce = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.cast(self.labels, tf.float32),
        #     logits=logits
        # )
        # loss = tf.reduce_mean(ce)
        #
        # prediction = tf.cast(tf.nn.sigmoid(logits) > threshold, tf.int32)
        #
        # accuracy = tf.cast(tf.equal(self.labels, prediction), tf.float32)
        #
        # self.out = {
        #     'prediction': prediction,
        #     'logits': tf.nn.sigmoid(logits),
        #     'accuracy': tf.reduce_mean(accuracy),
        #     'loss': loss
        # }
        #
        # self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)


        #regularization


        #
        prob=tf.nn.sigmoid(logits)
        #
        predi=tf.cast(prob>threshold,tf.int32)
        #
        accuracy=tf.cast(tf.equal(labels,predi),tf.float32)
        #
        #
        #
        self.out={'prediction':predi,'accuracy':tf.reduce_mean(accuracy),'loss':loss,'logits':logits}
        #
        self.train_op=tf.train.AdamOptimizer().minimize(loss)



def checking_model(model):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            out_put,_=sess.run([model.out,model.train_op],feed_dict={model.placeholder['input']:np.random.randint(0,10,[10,10]),model.placeholder['output']:np.random.randint(0,10,[10,28])})
            print(out_put['prediction'],out_put['accuracy'],out_put['loss'])




if "__main__"== __name__:

    bv=movies_classifier()
    checking_model(bv)















