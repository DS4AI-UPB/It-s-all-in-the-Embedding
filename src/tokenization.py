# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"


import re
import spacy
from stop_words import get_stop_words
from nltk.corpus import stopwords

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

def stopWordsEN():
    sw_stop_words = get_stop_words('en')
    sw_nltk = stopwords.words('english')
    sw_spacy = list(spacy.lang.en.stop_words.STOP_WORDS)
    sw_mallet = ['a', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards', 'again', 'against', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', 'course', 'currently', 'd', 'definitely', 'described', 'despite', 'did', 'different', 'do', 'does', 'doing', 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'happens', 'hardly', 'has', 'have', 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 'i', 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', 'way', 'we', 'welcome', 'well', 'went', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', 'wonder', 'would', 'would', 'x', 'y', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']
    return list(set(sw_stop_words + sw_nltk + sw_mallet + sw_spacy))

punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~-'
specialchar_re = re.compile('(%s)' % '|'.join(specialchar_dic.keys()))
nlp = spacy.load("en_core_web_sm")
cachedStopWords_en = stopWordsEN()

class Tokenization:
    def applyFE(self, text):
        """This method will combine the negation with the words
        Will result in a bigger vocabulary but with less bias
        """
        final_text = text.replace('cannot', 'can not')
        final_text = final_text.replace('can\'t', 'can not')
        final_text = final_text.replace('won\'t', 'will not')
        final_text = final_text.replace('n\'t', ' not')
        final_text = final_text.replace(' not ', ' not')

        return final_text

    def removeStopWords(self, text):
        return ' '.join([word for word in text.split() if word not in cachedStopWords_en])

    def removePunctuation(self, text, punctuation = punctuation):
        for c in punctuation:
            text = text.replace(c, ' ')
        return text

    def replaceUTF8Char(self, text, specialchars=specialchar_dic):
        def replace(match):
            return specialchars[match.group(0)]
        return specialchar_re.sub(replace, text)

    def createCorpus(self, text, remove_punctuation=True, remove_stopwords=True, apply_FE=True, id=None):
        corpus = []
        text_orig = text
        try:
            text = self.replaceUTF8Char(text).replace("\n", " ")
            doc = nlp(text)
            processed_text =  ' '.join([t.lemma_ if t.lemma_ != '-PRON-' else t.text if not t.ent_type_ else t.text for t in doc])
            processed_text = processed_text.replace("\s\s+", ' ')

            doc = nlp(processed_text.lower())

            rawText = not (remove_punctuation or remove_stopwords or apply_FE)

            for sentence in doc.sents:
                sent = str(sentence.text)
                if len(sent) == 0:
                    continue
                if not rawText:
                    if apply_FE:
                        sent = self.applyFE(text=sent)
                    if remove_punctuation:
                        sent = self.removePunctuation(text=sent)
                    if remove_stopwords:
                        sent = self.removeStopWords(text=sent)
                sent = sent.lower().split()
                if sent:
                    corpus.append(sent)
        except Exception as exp:
            print('exception =', str(exp))
            print('text =', text)
            print('text orig =', text_orig)
            print('ID =', id)
        # print(corpus)
        return corpus

    def __del__(self):
        print("Destructor Tokenization")

if __name__ == '__main__':
    tkn = Tokenization()
    text = "Apple data-intensive is looking at buying U.K. startup for $1 billion. This is great! The new D.P. model is funcitonal and ready"
    print(tkn.createCorpus(text))
    text = """The lion may be known as the king of the jungle, but lions do not live in jungles. They’re the rulers of the African savannahs that are covered in brown grasses and speckled with sparse trees. Lions’ coloring helps them blend in perfectly with the tall grass so they can ambush their prey as best as possible. And lions are ferocious. Although they’re one of the most powerful predators on land, lions are in danger. Hunters and poachers target lions to prove to the world their machismo.\n\nAnd while hunters seek to wipe lions off the face of the earth to bolster their egos, the Kevin Richardson Wildlife Sanctuary hopes to stop them and protect the big African cat at all cost.\n\nRichardson has earned the nickname the “Lion Whisperer” for a reason. He aims to educate the world about lions. And for those lucky enough to volunteer alongside Richardson, he encourages them to learn more about lions and help protect the wild species.\n\n“To raise awareness, Kevin has now set up his YouTube Channel ‘LionWhispererTV’. The channel is all about raising awareness about not only the declining numbers of lions but also how this rapid decrease is happening. By watching these videos, you are directly contributing to our scheme of land acquisition,” he writes in his bio.\n\nAs part of the volunteer program, Richardson hosts a “volunteer enrichment and lion enrichment” walk. As the name suggests, Richardson takes his group of volunteers out into the savannah of South Africa to hang out with two lions. There, the volunteers meet a male lion, Bobcat, and a female lioness, Gabby. Both lions look ferocious, but are truly “affectionate,” at least that’s what Richardson says. And remember, he’s the lion whisperer, so he’s got an advantage with these deadly big cats.\n\nAs Richardson showers the pair of lions with love, the volunteers stay locked in the truck, unwilling to put their lives in danger. And while they are in the vehicle, the lions are just feet from them – and if something goes wrong, they could wind up injured anyway.\n\nRichardson shared the video on his “The Lion Whisperer” YouTube channel. With more than one million hits, this video has proven to be one of his most famous.\n\nThe video describes the moment caught on tape as follows:\n\n“It’s an enrichment walk for both the volunteers and the lions as Kevin shows off his lovely lions as well as giving some amazing lion facts to the volunteers.”\n\nViewers like you are overwhelmed with the magnificent footage. The following are a few comments shared on the video.\n\n“I hope to someday volunteer there with Kevin. I believe in the work and his perspective about conservation. This video makes me want to all the more! Bobcat and Gabby are lovely lions.” “Every time I watch a one of your videos I somehow end up smiling from ear to ear!” “That was so beautiful, wish I could rub my head against a lion.”\n\nTake a moment to watch this video. Would you ever want to volunteer with Kevin Richardson and his lions?"""
    corpus = tkn.createCorpus(text, remove_stopwords=False)
    print(corpus)
