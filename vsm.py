import re
import numpy as np
from collections import defaultdict, OrderedDict, Counter
from math import log, sqrt
from os import listdir
from os.path import isfile, join

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class VectorSpaceModel():
    def __init__(self, docs):
        """
        The class is initialized with the documents stored as a dictionary, 
        with docIDs as keys and the list of terms as values.
        """
        self.docs = []
        for i, d in enumerate(docs):
            self.docs.append(self.preprocess_tokens(docs[i]))
        self.n_docs = len(self.docs)
        self.index = self.inverted_index()
        self.vocab = self.get_vocabulary()
        self.n_terms = len(self.vocab)
        self.tfidf = None


    def preprocess_tokens(self, tokens):
        """"
        Function to perform standard preprocessing.
        
        Given a list of tokens, it performs stop words removal and stemming, 
        and returns the list of clean tokens.
        """
        st = PorterStemmer()
        sw = stopwords.words('english')

        tokens = [word for word in tokens if word not in sw]
        tokens = [st.stem(word) for word in tokens]
        tokens = [word for word in tokens if word not in sw]
        return tokens


    def inverted_index(self):
        """
        Builds an inverted index.
        Returns a dictionary with terms as keys and for each term it stores 
        the set of docIDs as value, of the documents containing the term.
        """
        idx = dict()
        for docid in range(self.n_docs):  # for each document in the corpus
            for pos, t in enumerate(self.docs[docid]):  # for each term in the document
                idx[t] = idx.get(t, set())
                idx[t].add(docid)
        return idx


    def get_vocabulary(self):
        """"
        Returns the list of terms in the collection.
        """
        return list(self.index.keys())


    def inverse_doc_freq(self):
        """"
        Function to compute the inverse document frequency (IDF) for 
        each term in the collection.
        
        It returns a dictionary with terms as keys and the inverse document 
        frequency for the term as value.
        """
        idf = dict()
        for t in self.vocab:
            idf[t] = log( self.n_docs / len(self.index[t]) )
        return idf


    def docs_as_vectors(self):
        """
        Function to compute the Term Frequency - Inverse Document Frequency 
        (TF-IDF) for each term in each document of the collection.

        It returns a numpy array of shape (n_docs, n_terms), where each element
        tfidf[i,j] will contain the tfidf for the term j in document i.
        """
        self.tfidf = np.zeros((self.n_docs, self.n_terms))
        idf = self.inverse_doc_freq()

        for docid in range(self.n_docs):
            count = Counter(self.docs[docid])

            for t in self.vocab:
                if t in self.docs[docid]:
                    ind = self.vocab.index(t)
                    self.tfidf[docid][ind] = idf[t] * count[t]
        return self.tfidf


    def query_as_vector(self, query):
        """
        Function to convert a given query as a vector, where if the term is not 
        present in the query, the corresponding element will be 0, or 1 otherwise.

        It returns a numpy array, 0 or 1 in q_vec[i], depending on whether the term
        at index i is present or not in the query.
        """
        q = re.sub(r'[^a-zA-Z\s]+', '', query)
        q = self.preprocess_tokens(q.split())  # to return a list of terms
        q_vec = np.zeros(self.n_terms)

        for t in self.vocab:
            if t in q:
                ind = self.vocab.index(t)
                q_vec[ind] = 1
        return q_vec


    def relevance_scores(self, query):
        """"
        Function to compute the relevance score (as cosine similarity)
        between the given query and the documents in the collection.

        It returns a dictionary with docID as keys and the cosine similarity 
        sim(q,d) between the query and the corresponding document as values.
        """
        if self.tfidf is None:  # checks if the tfidf has already been computed
            self.tfidf = self.docs_as_vectors()

        if not isinstance(query, np.ndarray):  # checks if the query is already a vector
            query = self.query_as_vector(query)
        
        q_length = sqrt(sum(query**2))

        nonzero = np.where(query != 0)[0]
        scores = dict()  # for each document we store the cosine similarity between the query and the document
        for docid in range(len(self.docs)):
            d = self.tfidf[docid,]
            d_length = sqrt(sum(d**2))
            cos_sim = 0
            for idx in nonzero:
                cos_sim += (d[idx] * query[idx])
            
            if cos_sim == 0:
                scores[docid] = 0
            else:
                scores[docid] = cos_sim / (q_length * d_length)
        return scores


    def vector_space_model(self, query, k):
        """"
        Function to perform the ranked retrieval given a query, by computing
        the scores between the query and each document.

        It returns a dictionary with the k documents with the highest score. The keys 
        of the dictionary are the docIDs and the corresponding values are the computed score.
        """
        # first computes the relevance score given the query
        scores = self.relevance_scores(query)

        # sorts the results and returns the top k documents
        sorted_value = OrderedDict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        topk = {key : sorted_value[key] for key in list(sorted_value)[:k] if sorted_value[key]!=0}

        return topk


    def relevance_feedback_rocchio(self, query, rel_docs, nrel_docs=[], alpha=1, beta=.75, gamma=.15):
        """"
        Function implementing the Rocchio algorithm for the relevance feedback.

        It returns the optimized query vector obtained by applying the Rocchio model.
        """
        if self.tfidf is None:  # checks if the tfidf has already been computed
            self.tfidf = self.docs_as_vectors()
        
        if not isinstance(query, np.ndarray):  # checks if the query is already a vector
            query = self.query_as_vector(query)
        
        q_opt = np.zeros(self.n_terms)

        for t in self.vocab:
            idx = self.vocab.index(t)
            r = 0
            for docid in rel_docs:
                r += self.tfidf[docid,idx]
            r /= len(rel_docs)

            n = 0
            if len(nrel_docs) != 0:
                for docid in nrel_docs:
                    n += self.tfidf[docid,idx]
                n /= len(nrel_docs)
            else:
                gamma = 0  # if we do not have the list of non-relevand documents, it sets gamma to 0

            opt = alpha*query[idx] + beta*r - gamma*n
            if opt > 0:
                q_opt[idx] = opt
        return q_opt


    def pseudo_relevance_feedback(self, query, k=10):
        """"
        Function to implement the pseudo-relvance feedback.

        It first computes the scores to retrieve the k documents with highest score, 
        and then calls the function for the relevance feedback to compute the optimized query.
        """
        rel_docs = list(self.vector_space_model(query, k).keys())
        q_opt = self.relevance_feedback_rocchio(query, rel_docs, gamma=0)
        return q_opt


    def precision_recall(self, queries, relevance, min_k, max_k, rel_feedback=False, pseudo_feedback=False):
        """
        Function to compute precision and recall for each k in [min_k, max_k] for a set of 
        queries and relevance. It also allows to compute precision and recall for documents 
        retrieved adopting relevance and pseudo-relevance feedback, by setting the corresponding
        parameters to True.

        It returns two dictionaries, with values of k as keys and with numpy arrays as values,
        containing in each element i the precision and recall at k for query i.
        """
        ret = dict()
        rel = dict()
        for qid in list(queries.keys()):
            q = self.query_as_vector(queries[qid])
            rel[qid] = relevance[qid]
            if rel_feedback:
                nrel_docs = [i for i in range(self.n_docs) if i not in relevance[qid]]
                q = self.relevance_feedback_rocchio(q, relevance[qid], nrel_docs)
            if pseudo_feedback:
                q = self.pseudo_relevance_feedback(q)

            ret[qid] = self.vector_space_model(q, max_k)

        precision = dict()
        recall = dict()
        for k in range(min_k, max_k+1):
            precision[k] = np.zeros(len(queries.keys()))
            recall[k] = np.zeros(len(queries.keys()))
            for qid in list(queries.keys()):
                precision[k][qid-1] = len( set(list(ret[qid].keys())[:k]).intersection(set(rel[qid])) ) / len(list(ret[qid].keys())[:k])
                recall[k][qid-1] = len( set(list(ret[qid].keys())[:k]).intersection(set(rel[qid])) ) / len(rel[qid])

        return precision, recall



