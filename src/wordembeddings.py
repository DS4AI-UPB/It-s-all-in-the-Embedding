# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"


from glove import Corpus, Glove
from gensim.models import Word2Vec, FastText
from gensim import corpora
from mittens import Mittens, GloVe

import numpy as np

class WordEmbeddings:

    def __init__(self, corpus, normalize_tfidf=False):
        self.corpus = corpus
        self.normalize_tfidf = normalize_tfidf
        self.documents = []
        self.sentences = []
        self.word2id = {}
        self.no_words = 0
        self.max_size = 0 # max size of largest document
        self.no_docs = len(self.corpus)

    def preprareDocuments(self):
        word_id = 1
        for document in self.corpus:
            doc = []
            for sentence in document:
                self.sentences.append(sentence)
                for word in sentence:
                    if self.word2id.get(word) is None:
                        self.word2id[word] = word_id
                        word_id += 1
                    doc.append(self.word2id[word])
            if self.max_size < len(doc):
                self.max_size = len(doc)
            self.documents.append(doc)
        
        self.no_words = len(self.word2id) + 1
        
        return np.array(self.documents)
        

    def word2vecEmbedding(self, window_size=10, no_components=128, epochs=100, workers=4, sg=0, learning_rate=0.05):
        self.word2vec = np.empty(shape=(self.no_words, no_components))
        model = Word2Vec(self.sentences, vector_size=no_components, window=window_size, min_count=1, workers=workers, sg=sg, alpha=learning_rate, epochs=epochs)

        # for word in model.wv.vocab:
        #     print(word, model.wv[word])
        # print(model.wv['nefarious'])

        self.word2vec[0] = np.array([0] * no_components)
        for word in self.word2id:
            self.word2vec[self.word2id[word]] = np.array(model.wv[word])

        return self.word2vec

    
    def word2GloVeEmbedding(self, window_size=10, no_components=128, epochs=100, workers=4, learning_rate=0.05):
        self.word2glove = np.empty(shape=(self.no_words, no_components))
        corpus = Corpus() 
        #training the corpus to generate the co occurence matrix which is used in GloVe
        corpus.fit(self.sentences, window=window_size)
        # creating a Glove object which will use the matrix created in the above lines to create embeddings
        # We can set the learning rate as it uses Gradient Descent and number of components
        model = Glove(no_components=no_components, learning_rate=0.05)
        model.fit(corpus.matrix, epochs=epochs, no_threads=workers, verbose=False)
        # print(corpus.dictionary)
        model.add_dictionary(corpus.dictionary)
        # get the word vectors
        # for word in corpus.dictionary:
        #     print(word, model.word_vectors[corpus.dictionary[word]])
        # model.save('model.model')

        self.word2glove[0] = np.array([0] * no_components)
        for word in self.word2id:
            self.word2glove[self.word2id[word]] = model.word_vectors[corpus.dictionary[word]]

        return self.word2glove


    def word2FastTextEmbeddings(self, window_size=10, no_components=128, epochs=100, workers=4, sg=0, learning_rate=0.05):
        self.word2fasttext = np.empty(shape=(self.no_words, no_components))
        model = FastText(self.sentences, vector_size=no_components, window=window_size, min_count=1, workers=workers, sg=sg, alpha=learning_rate, epochs=epochs)


        # for word in model.wv.vocab:
        #     print(word, model.wv[word])
        # print(model.wv['nefarious'])
        self.word2fasttext[0] = np.array([0] * no_components)
        for word in self.word2id:
            self.word2fasttext[self.word2id[word]] = np.array(model.wv[word])

        return self.word2fasttext

    def word2MittensEmbedding(self, window_size=10, no_components=128, epochs=100, workers=4, learning_rate=0.05):
        self.word2mittens = np.empty(shape=(self.no_words, no_components))
        self.word2mittens[0] = np.array([0] * no_components)
        word2glove = {}
        corpus = Corpus() 
        #training the corpus to generate the co occurence matrix which is used in GloVe
        corpus.fit(self.sentences, window=window_size)
        # creating a Glove object which will use the matrix created in the above lines to create embeddings
        # We can set the learning rate as it uses Gradient Descent and number of components
        glove_model = Glove(no_components=no_components, learning_rate=0.05)
        glove_model.fit(corpus.matrix, epochs=epochs, no_threads=workers, verbose=False)
        # print(corpus.dictionary)
        glove_model.add_dictionary(corpus.dictionary)
        # get the word vectors
        # for word in corpus.dictionary:
        #     print(word, model.word_vectors[corpus.dictionary[word]])
        # model.save('model.model')

        vocabulary = []

        for word in self.word2id:
            word2glove[word] = glove_model.word_vectors[corpus.dictionary[word]]
            vocabulary.append(word)

        mittens_model = Mittens(n=no_components, max_iter=epochs)
        self.word2mittens[1:] = mittens_model.fit(corpus.matrix.toarray(), vocab=vocabulary, initial_embedding_dict=word2glove)

        return self.word2mittens

if __name__ == '__main__':
    corpus = [
        [
            ['Hello', 'this','tutorial', 'on', 'how','convert' ,'word',' integer','format'],
            ['this' ,'beautiful', 'day'],
            ['Jack','going' , 'office']
        ],
        [
            ['Hello', 'this','tutorial', 'on', 'how','convert' ,'word',' integer','format'],
            ['this' ,'beautiful', 'day'],
            ['Jack','going' , 'office']
        ],
        [
            ['Hello', 'this','tutorial', 'on', 'how','convert' ,'word',' integer','format'],
            ['this' ,'beautiful', 'day'],
            ['Jack','going' , 'office']
        ],
    ]

    we = WordEmbeddings(corpus)
    docs = we.preprareDocuments()
    print(docs.shape)
    print(docs)

    w2v = we.word2vecEmbedding()
    print(w2v)
    

    w2f = we.word2FastTextEmbeddings()
    print(w2f.shape)
    print(w2f)
    

    w2g = we.word2GloVeEmbedding()
    print(w2g.shape)
    print(w2g)
    

    w2m = we.word2MittensEmbedding()
    print(w2m.shape)
    print(w2m)
    
    print("\n\n")

    print(w2v[1])
    print(w2f[1])
    print(w2g[1])
    print(w2m[1])