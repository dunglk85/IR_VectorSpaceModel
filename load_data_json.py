import json


class Process():
    def __init__(self, corpus_file, queries_file, relevance_file):
        self.corpus = corpus_file
        self.queries = queries_file
        self.relevance = relevance_file
        self.docs = None
        self.queries = None
        self.relevance = None


    def load_docs(self):
        """
        Returns a list of lists where each sub-list contains all the terms 
        in the document at index i.
        The terms are not yet preprocessed through stemming or lemmatization.
        """
        
        with open(self.corpus, "r") as f:
            data = json.load(f)
            self.docs = [x['TEXT'] for x in data]
        


    def load_queries(self):
        """"
        Returns a dictionary of lists, with keys the queryID and as values 
        the query as a string in free-form text, removing punctuation.
        """
        with open(self.queries, "r") as f:
           data = json.load(f)
           self.queries = [x['QUERY'] for x in data['QUERIES']]


    def load_relevance(self):
        """"
        Returns a dictionary of lists, with keys the queryID and 
        as values the list of documents relevant to that query.
        """
        rel = dict()
        with open(self.relevance, "r") as f:
            t = []
            for row in f:
                r = row.split(" ")
                qid = int(r[0])
                rel[qid] = rel.get(qid, []) + [int(r[2])-1]
        self.relevance = rel

