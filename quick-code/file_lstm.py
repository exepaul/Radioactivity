import nltk
import sys
import numpy as np
import unicodedata




voca_data=np.load('final_vocab.npy').tolist() #vocabulary



punctuation=dict.fromkeys([i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')])

def remove_pun(text):
    #removing punctuation
    return [i.upper() for i in nltk.word_tokenize(text.translate(punctuation))]





vocab_dict={j:i for i,j in enumerate(voca_data)}

encoded_data=[]

def encode_query(text_):
    query=remove_pun(text_)
    sequence_data=[]
    for i in query:

        if i in vocab_dict:
            sequence_data.append(vocab_dict[i])

    return sequence_data





