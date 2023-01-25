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
from gensim.models import TfidfModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DocumentEmbeddings:

    def __init__(self, corpus, normalize_tfidf=False):
        self.corpus = corpus
        self.normalize_tfidf = normalize_tfidf

        self.documents = []
        self.sentences = []

        word_id = 0
        for document in self.corpus:
            doc = []
            for sentence in document:
                self.sentences.append(sentence)
                for word in sentence:
                    doc.append(word)
            self.documents.append(doc)
        if self.normalize_tfidf:
            self.tfidfVectorization()
        # print(self.documents)
        # print(self.sentences)


    def doc2VecEmbedding(self, window_size=10, no_components=128, epochs=100, workers=4, sg=0, learning_rate=0.05):
        self.doc2vec = []
        model = Word2Vec(self.sentences, vector_size=no_components, window=window_size, min_count=1, workers=workers, sg=sg, alpha=learning_rate, epochs=epochs)

        # for word in model.wv.vocab:
        #     print(word, model.wv[word])
        # print(model.wv['nefarious'])

        docidx = 0
        for document in self.documents:
            document_vector = np.array([0] * no_components)
            idx = 0
            for word in document:
                if self.normalize_tfidf:
                    document_vector = np.add(document_vector, self.tfidfs[docidx][word] * np.array(model.wv[word]))
                else:
                    document_vector = np.add(document_vector, np.array(model.wv[word]))
                idx += 1
            self.doc2vec.append(document_vector/idx)
            docidx += 1
        return np.array(self.doc2vec)

    def doc2GloVeEmbedding(self, window_size=10, no_components=128, epochs=100, workers=4, learning_rate=0.05):
        self.doc2glove = []
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
        docidx = 0
        for document in self.documents:
            document_vector = np.array([0] * no_components)
            idx = 0
            for word in document:
                if self.normalize_tfidf:
                    document_vector = np.add(document_vector, self.tfidfs[docidx][word] * np.array(model.word_vectors[corpus.dictionary[word]]))
                else:
                    document_vector = np.add(document_vector, np.array(model.word_vectors[corpus.dictionary[word]]))
                idx += 1
            self.doc2glove.append(document_vector/idx)
            docidx += 1
        return np.array(self.doc2glove)

    def doc2FastTextEmbeddings(self, window_size=10, no_components=128, epochs=100, workers=4, sg=0, learning_rate=0.05):
        self.doc2fasttext = []
        model = FastText(self.sentences, vector_size=no_components, window=window_size, min_count=1, workers=workers, sg=sg, alpha=learning_rate, epochs=epochs)

        # for word in model.wv.vocab:
        #     print(word, model.wv[word])
        # print(model.wv['nefarious'])
        
        docidx = 0
        for document in self.documents:
            document_vector = np.array([0] * no_components)
            idx = 0
            for word in document:
                if self.normalize_tfidf:
                    document_vector = np.add(document_vector, self.tfidfs[docidx][word] * np.array(model.wv[word]))
                else:
                    document_vector = np.add(document_vector, np.array(model.wv[word]))
                idx += 1
            self.doc2fasttext.append(document_vector/idx)
            docidx += 1
        return np.array(self.doc2fasttext)

    def tfidfVectorization(self, smartirs='atc'):
        dictionary = corpora.Dictionary(self.sentences)
        corpus = [ dictionary.doc2bow(document) for document in self.documents]
        tfidf = TfidfModel(corpus, smartirs=smartirs)
        self.tfidfs = []
        for document in tfidf[corpus]:
            doc = {}
            for id, freq in document:
                doc[dictionary[id]] = freq
            self.tfidfs.append(doc)
        return self.tfidfs

    def doc2TFIDF(self, min_df=4, max_features=5000):
        self.doc2tfidf = []
        corpus = [' '.join([sentence for sentence in sum(document, [])]) for document in self.corpus]
        tfidf = TfidfVectorizer(min_df=min_df, max_features=max_features)
        tfidf_fit = tfidf.fit(corpus)
        # vectorizing train data
        self.doc2tfidf = tfidf.transform(corpus).toarray()
        return np.array(self.doc2tfidf)


if __name__ == '__main__':
    corpus = [
        [
            ['Hello', 'this','tutorial', 'on', 'how','convert' ,'word',' integer','format'],
            ['this' ,'beautiful', 'day'],
            ['Jack','going' , 'office']
        ],
        [
            ['Hello', 'this','tutorial', 'on', 'how','convert' ,'word',' integer','format'],
            ['Jack','going' , 'office']
        ],
        [
            ['Hello', 'this','tutorial', 'on', 'how','convert' ,'word',' integer','format'],
            ['this' ,'beautiful', 'day'],
        ],
    ]

    we = DocumentEmbeddings(corpus)
    
    print("Doc2Vec Word2Vec CBOW")
    x= we.doc2VecEmbedding()
    print(x.shape)
    print("Doc2Vec Word2Vec SG")
    x= we.doc2VecEmbedding(sg=1)
    print(x.shape)
    print("Doc2Vec FastText CBOW")
    x= we.doc2FastTextEmbeddings()
    print(x.shape)
    print("Doc2Vec FastText SG")
    x= we.doc2FastTextEmbeddings(sg=1)
    print(x.shape)
    print("Doc2Vec GloVe")
    x= we.doc2GloVeEmbedding()
    print(x.shape)
    print("TFIDF")
    x= we.doc2TFIDF(min_df=1)
    print(x.shape)


    