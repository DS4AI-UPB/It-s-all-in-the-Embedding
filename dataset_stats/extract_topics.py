# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"


from vectorization import Vectorization
from topicmodeling import TopicModeling
import sys
import pandas as pd
import time

if __name__ == '__main__':
    FIN = sys.argv[1]
    num_topics = int(sys.argv[2])
    num_words = int(sys.argv[3])
    num_iterations = int(sys.argv[4])

    print("Start Read File!")
    df = pd.read_csv(FIN, encoding = "utf-8")
    # df.set_index(['id'], inplace=True)
    print("End Read File!")
    
    documents = []

    for text in df['content'].to_list():
        documents.append(text)

    print("Start Vectorization !")
    vec = Vectorization(documents)
    vec.vectorize()
    id2word = vec.getID2Word()
    corpus = vec.getTFIDFNorm()
    print("End Vectorization!")
    
    tm = TopicModeling(id2word=id2word, corpus=corpus)

    print("Start Topic Modeling NNF FULL!")
    print("=============NMF=============")

    topicsNMF = tm.topicsNMF(num_topics=num_topics, num_words=num_words, num_iterations=num_iterations)

    for topic in topicsNMF:
        print("TopicID", topic[0], topic[1], len(tm.topicDocsNMF[topic[0]]['docs']))

   
    print("End Topic Modeling NNF FULL!")
    print("\n\n=============================\n\n")

    print("\n\n=============================\n\n")
    print("Start Topic Modeling NNF by label!")
    docs_labels = {}
    labels = df['label'].unique()
    for label in labels:
        print("*****", label, "*****")
        documents = df.loc[df["label"] == label]['content'].to_list()

        print("Start Vectorization !")
        vec = Vectorization(documents)
        vec.vectorize()
        id2word = vec.getID2Word()
        corpus = vec.getTFIDFNorm()
        print("End Vectorization!")
    
        tm = TopicModeling(id2word=id2word, corpus=corpus)
        topicsNMF = tm.topicsNMF(num_topics=num_topics, num_words=num_words, num_iterations=num_iterations)

        for topic in topicsNMF:
            print("TopicID", topic[0], topic[1], len(tm.topicDocsNMF[topic[0]]['docs']))

    print("End Topic Modeling NNF by Label!")
    print("=============================")