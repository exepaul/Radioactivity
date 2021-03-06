import numpy as np
import tensorflow as tf
import random
sentences=np.load('encoded_sentences.npy')
labels=np.load('encoded_labels.npy')
# import movies_model

train_data=int(len(sentences)*0.85)

batch_size = 50
epoch=200
sequences=[i for i in zip(sentences,labels)]



train_data_final=sequences[:train_data]

test_data=sequences[train_data:]
iteration=int(len(train_data_final)//batch_size)
print(iteration)
np.random.shuffle(train_data_final)

def padding(vector_data):

    max_len=max([len(i) for i in vector_data])

    padded_sequences=[]

    for i in vector_data:
        if len(i)<max_len:
            padded_sequences.append(i+(max_len-len(i))*[0])

        else:
            padded_sequences.append(i)

    return padded_sequences



def evaluate(model):

    sess=tf.get_default_session()

    data_te=test_data


    accurac=[]

    for i in data_te:
        out=sess.run(model.out,feed_dict={model.placeholder['input']:[i[0]],model.placeholder['output']:[i[1]],model.placeholder['mode']:1})

        accurac.append(out['accuracy'])
        print(out['prediction'],'vs',[i[1]])


    return np.mean(accurac)


def train(model):

    sess=tf.get_default_session()

    for i in range(epoch):
        loss=[]

        for j in range(iteration):



            input_data=train_data_final[j*batch_size:(j+1)*batch_size]



            input_x=[]
            labels=[]

            for m in input_data:
                input_x.append(m[0])

                labels.append(m[1])


            input_x=padding(input_x)

            labels=labels

            out_put,_=sess.run([model.out,model.train_op],feed_dict={model.placeholder['input']:input_x,model.placeholder['output']:labels,model.placeholder['mode']:0})

            print("iterations",j,"loss",out_put['loss'],"accuracy",out_put['accuracy'])

        print("epoch",i,"accuracy",evaluate(model))

if "__main__" == __name__:

    model =movies_classifier()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pre_train=train(model)
