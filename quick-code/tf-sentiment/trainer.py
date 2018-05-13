import tensorflow as tf
import numpy as np
from reader import create_samples
from net import SentimentNetwork

from tqdm import tqdm

BATCH_SIZE=64
PAD=0
UNK=1


def seq_maxlen(seqs):
    """
    Maximum length of max-length sequence 
     in a batch of sequences
    Args:
        seqs : list of sequences
    Returns:
        length of the lengthiest sequence
    """
    return max([len(seq) for seq in seqs])

def pad_seq(seqs, maxlen=0, PAD=PAD, truncate=False):

    # pad sequence with PAD
    #  if seqs is a list of lists
    if type(seqs[0]) == type([]):

        # get maximum length of sequence
        maxlen = maxlen if maxlen else seq_maxlen(seqs)

        def pad_seq_(seq):
            if truncate and len(seq) > maxlen:
                # truncate sequence
                return seq[:maxlen]

            # return padded
            return seq + [PAD]*(maxlen-len(seq))

        seqs = [ pad_seq_(seq) for seq in seqs ]
    
    return seqs

def vectorize(samples):
    try:
        sentence = np.array(pad_seq([ s[0] for s in samples ]))
        pos      = np.array([ s[1] for s in samples ])
        label    = np.array([ s[2] for s in samples ])
    except:
        print('\n')
        print(samples)
        print('\n')
        input()
    return {
            'sentence' : sentence,
            'pos'      : pos,
            'label'    : label
            }

def train_run(netw, samples):
   sess = tf.get_default_session() 
   samples = vectorize(samples)
   return sess.run([ netw.train_op, netw.out ],
           feed_dict = {
               netw.placeholders['sentence'] : samples['sentence'],
               netw.placeholders['label'   ] : samples['label'   ],
               netw.placeholders['pos'   ]   : samples['pos'   ],
               netw.placeholders['mode'    ] : 0
               }
           )[1]

def evaluate(netw, testset, eval_batch_size=30):
    exec_g = lambda sample : sess.run(netw.out,
            feed_dict = {
                netw.placeholders['sentence'] : sample['sentence'],
                netw.placeholders['label'   ] : sample['label'   ],
                netw.placeholders['pos'   ]   : sample['pos'    ],
                netw.placeholders['mode'    ] : 1
                }
            )
    iterations = len(testset) // eval_batch_size
    return np.mean(np.array(
        [ exec_g(vectorize([testset[i]]))['accuracy'] 
            for i in tqdm(range(iterations)) ]
        ))


def train(netw, trainset, testset, epochs=100):

    iterations = len(trainset)//BATCH_SIZE
    for i in range(epochs):
        epoch_loss = []
        for j in tqdm(range(iterations)):
            out = train_run(netw, trainset[j * BATCH_SIZE : (j+1) * BATCH_SIZE])
            epoch_loss.append(out['loss'])

        # end of epoch
        print(i, 'loss', np.mean(np.array(epoch_loss)))
        print(i, 'accuracy', evaluate(netw, testset))


if __name__ == '__main__':
    vocab_size = 5000
    dataset = create_samples(max_vocab_size=vocab_size, consider_phrases=True)

    split_ = int(0.85 * len(dataset))
    trainset = sorted(dataset[:split_], key=lambda x : len(x[0]))
    testset  = sorted(dataset[split_:], key=lambda x : len(x[0]))

    print(vectorize(testset[:10]))

    """
    # create model
    netw = SentimentNetwork(vocab_size=vocab_size, hdim=50, wdim=50,
            dropout_value=0.5, lr=0.005)

    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       train(netw, trainset, testset)
    """
