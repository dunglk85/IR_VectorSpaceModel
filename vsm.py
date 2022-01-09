import re
from collections import defaultdict, OrderedDict, Counter
from math import log, sqrt
from os import listdir
from os.path import isfile, join


class VectorSpaceModel():
    def __init__(self, docs):
        """
        The class is initialized with the documents stored as a dictionary, 
        with docIDs as keys and the list of terms as values.
        """
        self.docs = docs
        self.n_docs = len(self.docs)
        self.index = self.inverted_index()
        self.vocab = self.get_vocabulary()
        self.n_terms = len(self.vocab)
        self.tfidf = None


    def get_vocabulary(self):
        """"
        Returns the list of terms in the collection.
        """
        return list(self.index.keys())


    def inverted_index(self):
        """
        Builds an inverted index.
        Returns a dictionary with terms as keys and for each term it stores 
        a list as value, with docIDs of the documents contianing the term.
        """
        idx = dict()
        for docid in list(self.docs.keys()):  # for each document in the corpus
            for pos, t in enumerate(self.docs[docid]):  # for each term in the document
                idx[t] = idx.get(t, set())
                idx[t].add(docid)
        return idx


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
        tfidf = np.zeros((self.n_docs, self.n_terms))
        idf = self.inverse_doc_freq()

        for docid in range(self.n_docs):
            count = Counter(self.docs[docid])

            for t in self.vocab:
                if t in self.docs[docid]:
                    ind = self.vocab.index(t)
                    tfidf[docid][ind] = idf[t] * count[t]
        return tfidf


    def query_as_vector(self, query):
        """
        Function to convert a given query as a vector, where if the term is not 
        present in the query, the corresponding element will be 0, or 1 otherwise.

        It returns a numpy array, 0 or 1 in q_vec[i], depending on whether the term
        at index i is present or not in the query.
        """
        q = self.process_query(query)  # to return a list of terms
        q_vec = np.zeros(n_terms)

        for t in self.vocab:
            if t in q:
                ind = vocab.index(t)
                q_vec[ind] = 1
        return q_vec


    def relevance_scores(self, query):
        """"
        Function to compute the relevance score (as cosine similarity)
        between the given query and the documents in the collection.

        It returns a dictionary with docID as keys and the cosine similarity 
        sim(q,d) between the query and the corresponding document as values.
        """
        scores = dict()  # for each document we store the cosine similarity between the query and the document
        query_terms = self.process_query(query)  # returns a list of the terms present in the query
        q = self.query_vector(query)
        q_length = sqrt(sum(q**2))

        if self.tfidf is None:
            self.tfidf = self.tfidf_vectors()

        for docid in range(len(self.docs)):
            d = tfidf[docid,]
            d_length = sqrt(sum(d**2))
            cos_sim = 0
            for t in query_terms:
                if t in self.vocab:
                    idx = self.vocab.index(t)
                    cos_sim += (d[idx] * q[idx])
            
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
        scores = self.relevance_scores(query)
        sorted_value = OrderedDict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        topk = {key : sorted_value[key] for key in list(sorted_value)[:k] if sorted_value[key]!=0}

        return topk


    def relevance_feedback_rocchio(self, query, rel_docs, nrel_docs=[], alpha=1, beta=.75, gamma=.15):
        """"
        Function implementing the Rocchio algorithm for the relevance feedback.

        It returns the optimized query vector.
        """
        if self.tfidf is None:
            self.tfidf = self.tfidf_vectors()
        
        query = self.query_as_vector(query)
        q_opt = np.zeros(self.n_terms)
        
        for t in self.vocab:
            idx = self.vocab.index(t)
            r = 0
            for docid in rel_docs:
                r += self.tfidf[docid,].sum()
            r /= len(rel_docs)

            if len(nrel_docs) != 0:
                n = 0
                for docid in nrel_docs:
                    n += self.tfidf[docid,].sum()
                n /= len(nrel_docs)
            else:
                gamma = 0

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
