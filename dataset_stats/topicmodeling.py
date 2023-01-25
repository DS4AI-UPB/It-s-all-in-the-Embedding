# coding: utf-8

__author__      = "Ciprian-Octavian Truică, Elena-Simona Apostol"
__copyright__   = "Copyright 2022, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "{ciprian.truica,elena.apostol}@upb.ro"
__status__      = "Development"


from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import numpy as np

class TopicModeling:
    def __init__(self, id2word, corpus, doc2class=None, num_cores=-1):
        self.id2word = id2word
        self.corpus = corpus        
        self.doc2class = doc2class
        self.num_cores = num_cores      
        self.doc2topicNMF = {}
        self.doc2topicKMeans = {}
        self.doc2topicLDA = {}
        self.topicDocsNMF = []
        self.topicDocsLDA = []
        self.topicDocsKMeans = []

 
    def topicsNMF(self, num_topics=10, num_words=10, num_iterations=2000, init="nndsvd", hardClustering=False, threshold=0.0):
        model = NMF(init=init, n_components=num_topics, max_iter=num_iterations)
        
        W = model.fit_transform(self.corpus)
        np.savetxt('NMF_' + str(num_topics) + 'topics_' + str(num_words) + 'words_'+ str(num_iterations) +'iter.csv', W, delimiter=",")
        H = model.components_
        tdNMF = {}
        for topic_idx in range(0, num_topics):
            tdNMF[topic_idx] = []
        # Documents for each topic and class
        if self.doc2class:
            doc_idx = 0
            for elem in W:
                topic_idx = np.where(elem==elem.max())[0][0]
                # make the dictionary
                if self.doc2topicNMF.get(self.doc2class[doc_idx]) is None:
                    self.doc2topicNMF[self.doc2class[doc_idx]] = {}
                    for i in range(0, num_topics):
                        self.doc2topicNMF[self.doc2class[doc_idx]][i] = 0
                self.doc2topicNMF[self.doc2class[doc_idx]][topic_idx] += 1
                tdNMF[topic_idx].append(doc_idx)
                doc_idx += 1
            # print(self.doc2topicNMF)
        # Documents for each topic
        else:
            doc_idx = 0
            for elem in W:
                if hardClustering:
                    topic_idx = np.where(elem==elem.max())[0][0]
                    tdNMF[topic_idx].append(doc_idx)
                else:
                    for value in elem:
                        if value > threshold:
                            topic_idx = np.where(elem==value)[0][0]
                            tdNMF[topic_idx].append(doc_idx)
                doc_idx += 1
        for elem in tdNMF:
            self.topicDocsNMF.append({"topic_id": elem, "docs": tdNMF[elem]})
        # NMF topics
        topics = []
        for topic_index in range( H.shape[0] ):
            top_indices = np.argsort( H[topic_index,:] )[::-1][0:num_words]
            term_ranking = [(self.id2word[i], H[topic_index][i]) for i in top_indices]
            topics.append((topic_index, term_ranking))
        return topics

    def topicsLDA(self, num_topics=10, num_words=10, num_iterations=2000, chunksize=20000, decay=0.5, learning_method='online', hardClustering=False, threshold=0.1):
        lda_model = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior=None, topic_word_prior=None, learning_method=learning_method, learning_decay=decay, learning_offset=10.0, max_iter=num_iterations, batch_size=chunksize, evaluate_every=-1, total_samples=chunksize, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=self.num_cores, verbose=0, random_state=None)
        lda_model.fit(self.corpus)
        W = lda_model.transform(self.corpus)
        np.savetxt('LDA_' + str(num_topics) + 'topics_' + str(num_words) + 'words_'+ str(num_iterations) +'iter_' + learning_method + '.csv', W, delimiter=",")
        H = lda_model.components_
        # print X
        tdLDA = {}
        for topic_idx in range(0, num_topics):
            tdLDA[topic_idx] = []
        if self.doc2class:
            doc_idx = 0
            for elem in W:
                topic_idx = np.where(elem==elem.max())[0][0]
                if self.doc2topicLDA.get(self.doc2class[doc_idx]) is None:
                    self.doc2topicLDA[self.doc2class[doc_idx]] = {}
                    for i in range(0, num_topics):
                        self.doc2topicLDA[self.doc2class[doc_idx]][i] = 0
                self.doc2topicLDA[self.doc2class[doc_idx]][topic_idx] += 1
                tdLDA[topic_idx].append(doc_idx)
                doc_idx += 1
            # print(self.doc2topicLDA)
        else:
            doc_idx = 0
            for elem in W:
                if hardClustering:
                    topic_idx = np.where(elem==elem.max())[0][0]
                    tdLDA[topic_idx].append(doc_idx)
                else:
                    for value in elem:
                        if value > threshold:
                            topic_idx = np.where(elem==value)[0][0]
                            tdLDA[topic_idx].append(doc_idx)
                doc_idx += 1

        for elem in tdLDA:
            self.topicDocsLDA.append({"topic_id": elem, "docs": tdLDA[elem]})

        topics = []
        for topic_index in range( H.shape[0] ):
            top_indices = np.argsort( H[topic_index,:] )[::-1][0:num_words]
            term_ranking = [(self.id2word[i], H[topic_index][i]) for i in top_indices]
            topics.append((topic_index, term_ranking))
        return topics

    def clustersKMeans(self, num_clusters=10, num_words=10, num_iterations=2000, n_init=10):
        kMeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=num_iterations, n_init=n_init, n_jobs=self.num_cores)
        kMeans.fit(self.corpus)
        order_centroids = kMeans.cluster_centers_.argsort()[:, ::-1]
        tdKMeans = {}
        for topic_idx in range(0, num_clusters):
            tdKMeans[topic_idx] = []
        if self.doc2class:
            doc_idx = 0
            for elem in kMeans.predict(self.corpus):
                if self.doc2topicKMeans.get(self.doc2class[doc_idx]) is None:
                    self.doc2topicKMeans[self.doc2class[doc_idx]] = {}
                    for i in range(0, num_clusters):
                        self.doc2topicKMeans[self.doc2class[doc_idx]][i] = 0
                self.doc2topicKMeans[self.doc2class[doc_idx]][elem] += 1
                # get the list of documents for the topic
                if tdKMeans.get(elem) is None:
                    tdKMeans[elem] = []
                tdKMeans[elem].append(doc_idx)
                doc_idx += 1
            # print(self.doc2topicKMeans)
        # Documents for each topic
        else:
            doc_idx = 0
            for elem in kMeans.predict(self.corpus):
                topic_idx = np.where(elem==elem.max())[0][0]
                if tdKMeans.get(topic_idx) is None:
                    tdKMeans[topic_idx] = []
                tdKMeans[topic_idx].append(doc_idx)
                doc_idx += 1
       
        for elem in tdKMeans:
            self.topicDocsKMeans.append({"topic_id": elem, "docs": tdKMeans[elem]})
                
        clusters  = []
        for i in range(num_clusters):
            terms = []
            for ind in order_centroids[i, :num_words]:
                terms.append((self.id2word[ind], order_centroids[i][ind]))
            clusters.append((i, terms)) 
        return clusters

    def __del__(self):
        print("Destructor TopicModeling")
