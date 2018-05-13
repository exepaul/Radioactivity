from nltk import word_tokenize
from nltk import FreqDist
from nltk import pos_tag
import spacy

from tqdm import tqdm

DATA = 'data/senticorpus.tsv'
PAD = 0
nlp = spacy.load('en')


def spacy_PoS(sentence):
    return [ w.pos_ for w in nlp(sentence) ]

def read_all(filename):
    samples = []
    with open(filename) as f:
        for line in f.readlines()[1:]:
            _, sent_id, sentence, sentiment = line.strip().split('\t')
            samples.append((sentence, sentiment))
    return samples

def read_sentences(filename):
    sent_dict = {}
    with open(filename) as f:
        for line in f.readlines()[1:]:
            _, sent_id, sentence, sentiment = line.strip().split('\t')

            if sent_id not in sent_dict:
                sent_dict[sent_id] = (sentence, sentiment)

            else:
                if len(sent_dict[sent_id][0]) < len(sentence):
                    sent_dict[sent_id] = (sentence, sentiment)

    return [ tuple(v) for k,v in sent_dict.items() ]

def build_vocabulary(samples, max_vocab_size):
    words = word_tokenize(' '.join([ text for text, senti in samples ]))
    # print('Total number of unique tokens : ', len(set(words)))
    fd = FreqDist(word_tokenize(' '.join([ text for text, senti in samples ])))
    return ['PAD', 'UNK' ] + [ w for w,f in fd.most_common(max_vocab_size) ]

def build_pos_vocabulary(samples):
    pos_vocab = []
    for sample in tqdm(samples):
        pos_vocab.extend(
                [ p for p in spacy_PoS(sample[0]) ]
                )
    return sorted(set(pos_vocab))

def index_samples(samples, vocab, pos_vocab):
    w2i = { w:i for i,w in enumerate(vocab) }
    w2i_ = lambda w : w2i[w] if w in w2i else 1
    p2i = { p:i for i,p in enumerate(pos_vocab) }

    indexed_samples = []
    for sentence, sentiment in tqdm(samples):
        tokenized = [ w for w in word_tokenize(sentence) ]
        # PoS tag
        pos = [ p for p in spacy_PoS(tokenized) ]
        indexed_samples.append( 
                ([ w2i_(w) for w in tokenized ],
                 [ p2i[p]  for p in spacy_PoS(sentence) ],
                 int(sentiment)) 
                )

    #return sorted(indexed_samples, 
    #        key = lambda x : len(x[0]),
    #        reverse=True
    #        )
    shuffle(index_samples)
    return index_samples

def create_samples(max_vocab_size, consider_phrases=True):
    samples = read_all(DATA) if consider_phrases else read_sentences(DATA)
    vocab = build_vocabulary(samples, max_vocab_size)
    return index_samples(samples, vocab, build_pos_vocabulary(samples))
