# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"


import os
import sys

# helpers
import time

# classification
import numpy as np
import pandas as pd
import sys
import os
import random as rnd
import math
from scipy import io as sio
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from transformers import TFBertModel, TFRobertaModel, TFBartModel
from transformers import BertTokenizer, RobertaTokenizer, BartTokenizer

from simpletransformers.language_representation import RepresentationModel
from sentence_transformers import SentenceTransformer

from documentembeddings import DocumentEmbeddings
from tokenization import Tokenization

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"


tkn = Tokenization()

def processElement(elem):
    id_line = elem[0]
    text = elem[1]
    text = tkn.createCorpus(text, remove_stopwords=False, apply_FE=False, id=id_line)
    return id_line, text

def getBERTEncoding(X, use_cuda=False):
    model = RepresentationModel(model_type="bert", model_name="bert-large-uncased", use_cuda=use_cuda)

    X_BERT = model.encode_sentences(X, combine_strategy="mean")
    print("BERT pre-reshare")
    print(X_BERT)

    return X_BERT

def getRoBERTaEncoding(X, use_cuda=False):
    model = RepresentationModel(model_type="roberta", model_name="roberta-large", use_cuda=use_cuda)
    X_RoBERTa = model.encode_sentences(X, combine_strategy="mean")
    print("RoBERTa pre-reshare")
    print(X_RoBERTa)

    return X_RoBERTa

def getBERTEncodingST(X):
    model = SentenceTransformer('bert-large-uncased')
    X_BERT = model.encode(X)
    print("BERT pre-reshare")
    print(X_BERT)
    return X_BERT

def getRoBERTaEncodingST(X):
    model = SentenceTransformer('roberta-large')
    X_RoBERTa = model.encode(X)
    print("RoBERTa pre-reshare")
    print(X_RoBERTa)

    return X_RoBERTa

def getBARTEncoding(X):
    model = SentenceTransformer('facebook/bart-large')
    X_BART = model.encode(X)
    print("BART pre-reshare")
    print(X_BART)

    return X_BART



if __name__ =="__main__":
    FIN = sys.argv[1]
    DIREMBS = sys.argv[2] # the directory
    USE_CUDA = bool(int(sys.argv[3])) # 0 - False, 1 - True
    print(USE_CUDA)

    dataSet = pd.read_csv(FIN, encoding = "utf-8")
    labels = dataSet['label'].unique()
    print("No. classes", labels)
    dataSet.dropna(inplace=True)
    dataSet.reset_index(drop=True, inplace=True)
    dataSet['id'] = dataSet.index

    X = dataSet['content'].astype(str).to_list()
    y = dataSet['label'].astype(int).to_list()
    sio.savemat(os.path.join(DIREMBS, 'labels.mat'), {'y': y})


    start_time = time.time()
    X_BERT = getBERTEncodingST(X)
    end_time = time.time()
    print("Time taken extract BERT Embeddings: ", end_time - start_time)

    start_time = time.time()
    X_BART = getBARTEncoding(X)
    end_time = time.time()
    print("Time taken extract BART Embeddings: ", end_time - start_time)

    start_time = time.time()
    X_RoBERTa = getRoBERTaEncoding(X)
    end_time = time.time()
    print("Time taken extract RoBARTa Embeddings: ", end_time - start_time)

    sio.savemat(os.path.join(DIREMBS, 'D2V_BERT.mat'), {'X': X_BERT})
    sio.savemat(os.path.join(DIREMBS, 'D2V_BART.mat'), {'X': X_BART})
    sio.savemat(os.path.join(DIREMBS, 'D2V_RoBERTa.mat'), {'X': X_RoBERTa})

    print("Start Tokenization")
    texts = dataSet[['id', 'content']].to_numpy().tolist()

    corpus = [None] * len(texts)
    no_threads = cpu_count() - 1
    with ProcessPoolExecutor(max_workers=no_threads) as worker:
        for result in worker.map(processElement, texts):
            if result:
                corpus[result[0]] = result[1]

    print("End Tokenization")

    we = DocumentEmbeddings(corpus)

    print("Doc2Vec Word2Vec CBOW")
    X_D2V_W2V_CBOW = we.doc2VecEmbedding()
    sio.savemat(os.path.join(DIREMBS, 'D2V_W2V_CBOW.mat'), {'X': X_D2V_W2V_CBOW})

    print("Doc2Vec Word2Vec SG")
    X_D2V_W2V_SG = we.doc2VecEmbedding(sg=1)
    sio.savemat(os.path.join(DIREMBS, 'D2V_W2V_SG.mat'), {'X': X_D2V_W2V_SG})

    print("Doc2Vec FastText CBOW")
    X_D2V_FT_CBOW = we.doc2FastTextEmbeddings()
    sio.savemat(os.path.join(DIREMBS, 'D2V_FT_CBOW.mat'), {'X': X_D2V_FT_CBOW})

    print("Doc2Vec FastText SG")
    X_D2V_FT_SG = we.doc2FastTextEmbeddings(sg=1)
    sio.savemat(os.path.join(DIREMBS, 'D2V_FT_SG.mat'), {'X': X_D2V_FT_SG})

    print("Doc2Vec GloVe")
    X_D2V_GLOVE = we.doc2GloVeEmbedding()
    sio.savemat(os.path.join(DIREMBS, 'D2V_GLOVE.mat'), {'X': X_D2V_GLOVE})

    print("TFIDF")
    X_D2V_TFIDF = we.doc2TFIDF(min_df=5)
    sio.savemat(os.path.join(DIREMBS, 'D2V_TFIDF.mat'), {'X': X_D2V_TFIDF})

