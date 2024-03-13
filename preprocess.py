import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds, eigsh

class VectorSpaceModel():
    def __init__(self, docs):
        """
        The class is initialized with the documents stored as a dictionary, 
        with docIDs as keys and the list of terms as values.
        """
        self.A = docs
        m, n = docs.shape
        self.left = m < n
        self.alpha = None
        self.beta = None
        self.lanczos_vectors = None
        self.norms = None
        self.q1 = None
        self.scores = None
        self.len = min(m, n)
        self.n = n
        v = np.random.rand(self.len)
        self.q1 = v / np.linalg.norm(v)


    def preprocess(self, k, tridiag=False):
        self.k = k
        self.tridiag = tridiag
        self.reortho = {}
        A = self.A
        len = self.len
        n = self.n
        epsilon = np.sqrt(np.finfo(np.float64).eps)
        q = [np.zeros(len) for i in range(self.k)]
        a = np.zeros(n)
        beta = np.zeros(self.k)
        alpha = np.zeros(self.k)
        q[0] = self.q1
        memo = {}
        count = 0
        for i in range(self.k-1):
            if self.left:
                q_hat = A.T.dot(q[i])
                w = A.dot(q_hat) - beta[i] * q[i-1]
                alpha[i] = w.dot(q[i])
                a = a + np.square(q_hat)
            else:
                q_hat = A.dot(q[i])
                w = A.T.dot(q_hat) - beta[i] * q[i-1]
                alpha[i] = w.dot(q[i])
                a = a + alpha[i]*np.square(q[i]) + 2*beta[i]*np.multiply(q[i], q[i-1])
            
            w = w - alpha[i] * q[i]
            ## start full reorthonization
            # for j in range(i):
            #     w_dotq = w @ q[j]
            #     w = w - w_dotq * q[j]
            
            # start partial reorthonization
            bound = np.linalg.norm(w) * epsilon
            for j in range(i):
                w_dotq = w @ q[j]
                if abs(w_dotq) > bound:
                    self.reortho[(i,j)] = w_dotq
                    w = w - w_dotq * q[j]
            # end partial reorthonization

            beta[i+1] = np.linalg.norm(w)
            if beta[i+1] == 0:
                self.k = i
                break
            q[i+1] = w / beta[i+1]

        self.alpha = alpha
        self.beta = beta
        self.q1 = q[0]
        self.lanczos_vectors = q

    
    def response(self, query):
        s_hat = np.zeros(self.len)
        if self.left:
            s_hat = query
        else:
            s_hat = self.A.T.dot(query)

        s = np.zeros(self.len)
        if self.tridiag:
            q = [np.zeros(self.len) for _ in range(self.k)]
            q[0] = self.q1
        else:
            q = self.lanczos_vectors
        
        for i in range(self.k-1):
            q_dot_query = q[i] @ s_hat
            s = s + q_dot_query * q[i]

            if self.tridiag:
                if self.left:
                    Aq = self.A.T.dot(q[i])
                    w = self.A.dot(Aq)
                else:
                    Aq = self.A.dot(q[i])
                    w = self.A.T.dot(Aq)

                w = w - self.beta[i] * q[i-1] - self.alpha[i] * q[i]
                for j in range(i):
                    if (i, j) in self.reortho:
                        if self.reortho[(i,j)] == 0:
                            break
                        w = w - self.reortho[(i,j)] * q[j]
                q[i+1] = w / self.beta[i+1]
                 
        q_dot_query = q[self.k-1] @ s_hat
        s = s + q_dot_query * q[self.k-1]

        if self.left:
            self.scores = self.A.T.dot(s)
        else:
            self.scores = s


    def lsi_preprocess(self, k):
        u, s, vt = svds(self.A,k=k)
        row_norms = np.linalg.norm(vt.T * s, axis=1)
        return u, s, vt, row_norms
    
    def lsi_response(self, u, s, vt, query):
        ndot = query.T.dot(u)
        return ndot.dot(np.diag(s)).dot(vt)
    

    import numpy as np

    def _svd(self):
        T = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
        eigenvalues, eigenvectors = eigsh(T, k=k)
        return eigenvalues, eigenvectors

    def svd_response(self, eigenvectors, query):
        if self.left:
            pass

    