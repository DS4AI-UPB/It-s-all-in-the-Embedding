# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"


import pandas as pd
import re
import spacy
import sys
from nltk.corpus import stopwords, words
from stop_words import get_stop_words
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

nlp = spacy.load('en_core_web_lg')

def stopWordsEN():
    sw_stop_words = get_stop_words('en')
    sw_nltk = stopwords.words('english')
    sw_spacy = list(spacy.lang.en.stop_words.STOP_WORDS)
    sw_mallet = ['a', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', 'different', 'do', 'does', 'doing', 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'happens', 'hardly', 'has', 'have', 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', 'way', 'we', 'welcome', 'well', 'went', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', 'wonder', 'would', 'would', 'x', 'y', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']
    return list(set(sw_stop_words + sw_nltk + sw_mallet + sw_spacy))

cachedStopWords_en = stopWordsEN()
wordsSet = set(words.words())
punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~-'

def processElement(row):
    docid = int(row[0])
    doc = nlp(row[1])
    document = [token.lemma_ for token in doc if token.lemma_ != "-PRON-" and not token.lemma_ in stop_words]
    label = row[2]
    tokens = word_tokenize(str(row[0]))
    unigrams = []
    for elem in list(ngrams(tokens,1)):
        if (elem[0][0] in punctuation or elem[0][-1] in punctuation) or (elem[0].lower() in cachedStopWords_en):
            continue
        elif elem[0] in wordsSet and (len(elem[0]) >= 3 or elem[0] == 'no'):
            unigrams.append(elem)
    return [docid, document, len(document), unigrams, label]

if __name__ =="__main__":
    FIN = sys.argv[1]
    FOUT = sys.argv[2]
    stop_words = set(stopwords.words('english'))
    no_threads = cpu_count()

    pattern = re.compile("\.\.+")

    data = pd.read_csv(FIN, encoding = "utf-8")
    
    newds = []
    
    labels = data['label'].unique()
    unigrams = {}
    for label in labels:
        unigrams[label] = []

    with ProcessPoolExecutor(max_workers=no_threads) as worker:
        for result in worker.map(processElement, data.to_numpy()):
            if result:
                newds.append(result)
                unigrams[result[4]] += result[3]

    for label in labels:
        unigrams_top10 = Counter(unigrams[label]).most_common(10)
        print(label, unigrams_top10)

    dataset = pd.DataFrame(newds, columns = ['id', 'content', 'doclen', 'unigrams', 'label'])
    dataset.to_csv(FOUT, encoding = "utf-8", index = False, header=True)
    result = dataset.groupby('label').agg({'doclen': ['mean', 'min', 'max', 'std']}) 
    
    print(result)
    print(dataset.describe())

    texts = {}
    total_total = []
    for label in dataset.label.unique():
        texts[label] = dataset.loc[dataset["label"] == label, "content"].tolist()
        text_label = sum(texts[label], [])
        texts[label] = text_label
        total_total += text_label
        print(label, "tokens", len(text_label))
        print(label, "tokens unique", len(set(text_label)))
    
    print("total tokens", len(total_total))
    print("total tokens unique", len(set(total_total)))


