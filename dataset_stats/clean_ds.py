# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"

import numpy as np
import pandas as pd
import sys
import time
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from nltk.corpus import stopwords
import spacy

specialchar_dic={
    "’": "'",
    "„": "\"",
    "“": "\"",
    "”": "\"",
    "«": "<<",
    "»": ">>",
    "…": "...",
    "—": "--",
    "¡": "!",
    "¿": "?",
    "©": " ",
    "–": " "
}

specialchar_re = re.compile('(%s)' % '|'.join(specialchar_dic.keys()))

punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~-'

stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_lg')

no_threads = cpu_count()


def removePunctuation(text, punctuation = punctuation):
    for c in punctuation:
        text = text.replace(c, ' ')
    return text

def replaceUTF8Char(text, specialchars=specialchar_dic):
    def replace(match):
        return specialchars[match.group(0)]
    return specialchar_re.sub(replace, text)

def processElement(row):
    clean_text = str(row[1])
    clean_text = replaceUTF8Char(clean_text)
    clean_text = clean_text.replace("-"," ").replace('[^\w\s]','').replace('\n',' ')
    doc = nlp(clean_text)
    clean_text = ' '.join([token.lemma_ for token in doc if token.lemma_ != "-PRON-" and not token.lemma_ in stop_words])
    clean_text = removePunctuation(clean_text)
    clean_text = clean_text.lower()
    clean_text = re.sub(' +', ' ', clean_text)
    
    return [row[0], clean_text, row[2]]


if __name__ == "__main__":
    FILE_NAME = sys.argv[1]

    dataSet = pd.read_csv(FILE_NAME, encoding = "utf-8") 

    
    clean_texts = []
    with ProcessPoolExecutor(max_workers=no_threads) as worker:
        for result in worker.map(processElement, dataSet.to_numpy()):
            if result:
                clean_texts.append(result)


    df = pd.DataFrame(clean_texts, columns = ['id', 'content', 'labels'])
    df.sort_values(by=['id'], inplace=True)
    df.to_csv('corpus_clean.csv', index=False)