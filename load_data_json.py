import json


class LoadDataset():
    def __init__(self, corpus_file, queries_file, relevance_file):
        self.corpus = corpus_file
        self.query_file = queries_file
        self.relevance_file = relevance_file
        self.docs = None
        self.queries = None
        self.relevances = None


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
        with open(self.query_file, "r") as f:
           data = json.load(f)
           self.queries = [x['QUERY'] for x in data['QUERIES']]


    def load_relevance(self):
        """"
        Returns a dictionary of lists, with keys the queryID and 
        as values the list of documents relevant to that query.
        """
        rel = dict()
        with open(self.relevance_file, "r") as f:
            t = []
            for row in f:
                r = row.split(" ")
                qid = int(r[0])
                rel[qid] = rel.get(qid, []) + [int(r[2])-1]
        self.relevance = rel

