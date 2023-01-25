# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize


class Vectorization:
    def __init__(self, corpus):
        self.corpus = corpus

    def vectorize(self, max_features=5000, smooth_idf=True, normalize_tfidf=True, norm='l2'):
        self.vectorizer = CountVectorizer(analyzer='word', max_features=max_features)
        self.counts = self.vectorizer.fit_transform(self.corpus)
        transformer = TfidfTransformer(smooth_idf=smooth_idf)
        self.tfidf = transformer.fit_transform(self.counts)
        if normalize_tfidf:
            self.tfidf_norm = normalize(self.tfidf, norm=norm, axis=1)

    def getCount(self):
        return self.counts

    def getTFIDF(self):
        return self.tfidf

    def getTFIDFNorm(self):
        return self.tfidf_norm

    def getID2Word(self):
        self.id2word = {}
        idx = 0
        for elem in self.vectorizer.get_feature_names():
            self.id2word[idx] = elem
            idx += 1
        return self.id2word

    def __del__(self):
        print("Destructor Vectorization")
