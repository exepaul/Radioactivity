import numpy as np
import random
sentences=np.load('sentences.npy')
labels=np.load('labelsa.npy')


train_data=int(len(sentences)*0.85)
print(train_data)

batch_size = 2
iteration=int(len(sentences)//batch_size)


def get_train_data():

    data=sentences[:train_data]
    labels_data= labels[:train_data]

    values_batch=np.random.randint(0,len(sentences)-batch_size)

    print(values_batch,values_batch+batch_size)

    batch_data=data[values_batch:values_batch+batch_size]

    labels_data = labels_data[values_batch:values_batch+batch_size]

    return {'batch':batch_data,
            'labels':labels_data

            }





print(get_train_data())
