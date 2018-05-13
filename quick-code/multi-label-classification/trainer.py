from reader import struc_data
from network import *

import tensorflow as tf
import numpy as np
from random import shuffle

BATCH_SIZE=10


def evaluate(model, testset):
    sess = tf.get_default_session()
    return np.mean(np.array([ sess.run(model.out['accuracy'],
        feed_dict = {
            model.attrs  : np.array(attrs).reshape([1, 103]),
            model.labels : np.array(labels).reshape([1, 14])
            }) for attrs, labels in testset] ))

def predict(model, sample):
    sess = tf.get_default_session()
    return sess.run(model.out['prediction'],
            feed_dict = { 
                model.attrs  : np.array(sample[0]).reshape([1, 103]), 
                model.labels : np.array(sample[1]).reshape([1, 14])
                })
 

def train(model, trainset, testset, epochs=20):

    iterations = len(trainset) // BATCH_SIZE

    sess = tf.get_default_session()
    for i in range(epochs):
        losses = []
        for j in range(iterations):
            batch = trainset[ j * BATCH_SIZE : (j+1) * BATCH_SIZE ]
            batch_attrs =  [ np.array(item) for item, b in batch ]
            batch_labels = [ np.array(item) for a, item in batch ]
            _, out = sess.run([model.train_op, model.out],
                    feed_dict = {
                        model.attrs  : batch_attrs,
                        model.labels : batch_labels
                        })
            losses.append(out['loss'])

        accuracy = evaluate(model, testset)
        print('epoch {} : \n\tloss : {}\n\taccuracy : {}'.format(
            i, np.mean(np.array(losses)), accuracy))


if __name__ == '__main__':
    # get dataset
    data = struc_data()
    shuffle(data)
    split_ = int(0.85*len(data))
    trainset = data[:split_]
    testset  = data[split_:]

    # create model
    #model = LogisticRegressor(103, 14, lr=0.1)
    model = MLP(103, 14, hdim=100, lr=0.05)


    # train and eval
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(model, trainset, testset, epochs=1000)
        for sample in testset[350:380]:
            print(sample[1], predict(model, sample))
