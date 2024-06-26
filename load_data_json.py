import json
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class LoadDataset():
    def __init__(self, corpus_file, queries_file, relevance_file, name):
        self.corpus = corpus_file
        self.query_file = queries_file
        self.relevance_file = relevance_file
        self.doc_matrix = None
        self.query_vectors = None
        self._load_docs()
        self._load_queries()
        self.relevance = self._load_relevance()
        self.name = name
        

    def tokenize_and_preprocess(self, doc):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(doc.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
        return tokens

    def _load_docs(self):
        """
        Returns a list of lists where each sub-list contains all the terms 
        in the document at index i.
        The terms are not yet preprocessed through stemming or lemmatization.
        """
        
        with open(self.corpus, "r") as f:
            data = json.load(f)
            self.docs = [x['TEXT'] for x in data]
            self.vectorizer = CountVectorizer(tokenizer=self.tokenize_and_preprocess)
            X = self.vectorizer.fit_transform(self.docs)
            self.tfidf_transformer = TfidfTransformer()
            self.doc_matrix = self.tfidf_transformer.fit_transform(X)
            self.doc_matrix = self.doc_matrix.transpose()
        


    def _load_queries(self):
        """"
        Returns a dictionary of lists, with keys the queryID and as values 
        the query as a string in free-form text, removing punctuation.
        """
        with open(self.query_file, "r") as f:
           data = json.load(f)
           self.queries = [x['QUERY'] for x in data['QUERIES']]
        if self.doc_matrix == None:
            self._load_docs()
        X = self.vectorizer.transform(self.queries)
        self.query_vectors = self.tfidf_transformer.transform(X)
        self.query_vectors = self.query_vectors.transpose()


    def _load_relevance(self):
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
        return rel

        
    


