import re


class LoadDataset():
    def __init__(self, corpus_file, queries_file, relevance_file):
        self.corpus = corpus_file
        self.queries = queries_file
        self.relevance = relevance_file

    def load_docs(self):
        """
        Returns a list of lists where each sub-list contains 
        all the terms in the document at index i.
        """
        arts = []
        with open(self.corpus, "r") as f:
            t = []
            r = False
            for row in f:
                if row.startswith(".W"):
                    r = True
                    continue
                if row.startswith(".I"):
                    if t != []:
                        arts.append(t)
                    t = []
                    r = False
                if r:
                    row = re.sub(r'[^a-zA-Z\s]+', '', row)
                    t += row.split()
        return arts

    def load_queries(self):
        """"
        Returns a dictionary of lists, with keys the queryID 
        and as values a list of terms occurring in the query.
        """
        q = dict()
        with open(self.queries, "r") as f:
            t = []
            r = False
            for row in f:
                if row.startswith(".W"):
                    r = True
                    continue
                if row.startswith(".I"):
                    if t != []:
                        q[qid] = t
                    t = []
                    r = False
                    qid = int(row[3:].replace("\n", ""))
                if r:
                    row = re.sub(r'[^a-zA-Z\s]+', '', row)
                    t += row.split()
        return q

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
                rel[qid] = rel.get(qid, []) + [int(r[2])]

        return rel
