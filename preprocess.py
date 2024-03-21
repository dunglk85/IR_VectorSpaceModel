import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds, eigsh
import math

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
        q = np.zeros((len, k))
        a = np.zeros(n)
        beta = np.zeros(self.k)
        alpha = np.zeros(self.k)
        q[:,0] = self.q1
        memo = {}
        count = 0
        for i in range(self.k-1):
            if self.left:
                q_hat = A.T.dot(q[:,i])
                w = A.dot(q_hat) - beta[i] * q[:,i-1]
                alpha[i] = w.dot(q[:,i])
                a = a + np.square(q_hat)
            else:
                q_hat = A.dot(q[:,i])
                w = A.T.dot(q_hat) - beta[i] * q[:,i-1]
                alpha[i] = w.dot(q[:,i])
                a = a + alpha[i]*np.square(q[:,i]) + 2*beta[i]*np.multiply(q[:,i], q[:,i-1])
            
            w = w - alpha[i] * q[:,i]
            ## start full reorthonization
            # for j in range(i):
            #     w_dotq = w @ q[j]
            #     w = w - w_dotq * q[j]
            
            # start partial reorthonization
            bound = np.linalg.norm(w) * epsilon
            for j in range(i):
                w_dotq = w @ q[:,j]
                if abs(w_dotq) > bound:
                    self.reortho[(i,j)] = w_dotq
                    w = w - w_dotq * q[:,j]
            # end partial reorthonization

            beta[i+1] = np.linalg.norm(w)
            if beta[i+1] == 0:
                self.k = i
                break
            q[:,i+1] = w / beta[i+1]

        self.alpha = alpha
        self.beta = beta
        self.q1 = q[:,0]
        self.lanczos_vectors = q
        self.norms = a

    
    def response(self, query):
        s_hat = np.zeros(self.len)
        if self.left:
            s_hat = query
        else:
            s_hat = self.A.T.dot(query)

        s = np.zeros(self.len)
        if self.tridiag:
            q = np.zeros((self.len, self.k))
            q[:,0] = self.q1
        else:
            q = self.lanczos_vectors
        
        for i in range(self.k-1):
            q_dot_query = q[:,i] @ s_hat
            s = s + q_dot_query * q[:,i]

            if self.tridiag:
                if self.left:
                    Aq = self.A.T.dot(q[:,i])
                    w = self.A.dot(Aq)
                else:
                    Aq = self.A.dot(q[:,i])
                    w = self.A.T.dot(Aq)

                w = w - self.beta[i] * q[:,i-1] - self.alpha[i] * q[:,i]
                for j in range(i):
                    if (i, j) in self.reortho:
                        if self.reortho[(i,j)] == 0:
                            break
                        w = w - self.reortho[(i,j)] * q[:,j]
                q[:,i+1] = w / self.beta[i+1]
                 
        q_dot_query = q[:,self.k-1] @ s_hat
        s = s + q_dot_query * q[:,self.k-1]

        if self.left:
            self.scores = self.A.T.dot(s)
        else:
            self.scores = s


    def implicit_qr_algorithm(self, alpha, beta, eigenvectors=None, tolerance=1e-10):
        n = len(alpha)
        if n < 3:
            return alpha, eigenvectors
        if np.all(eigenvectors == None):
            eigenvectors = np.eye(n)

        for _ in range(100):
            # Perform implicit QR step
            d = (alpha[n-2] - alpha[n-1])/2
            sign_d = -1 if d < 0 else 1
            mu = alpha[n-1] - beta[n-1]*beta[n-1]/(d + sign_d*np.sqrt(d*d + beta[n-1]*beta[n-1]))
            x = alpha[0] - mu
            z = beta[1]
            for i in range(n - 1):
                # Compute the Givens rotation
                
                theta = np.arctan(-z / x)
                c = np.cos(theta)
                s = np.sin(theta)

                eigenvectors[:, [i, i + 1]] = eigenvectors[:, [i, i + 1]] @ np.array([[c, s], [-s, c]])
                
                beta[i] = c*beta[i] - s*z
                if i < n - 2:
                    z = -s * beta[i+2]
                    beta[i+2] = c*beta[i+2]

                tem_a_1 = alpha[i]
                tem_a_2 = alpha[i+1]
                alpha[i] = c*c * tem_a_1 - 2*c*s * beta[i+1] + s*s*tem_a_2
                alpha[i+1] = s*s * tem_a_1 + 2*c*s * beta[i+1] + c*c*tem_a_2
                beta[i+1] = s*c*(tem_a_1 - tem_a_2) + beta[i+1]*(c*c - s*s)

                if abs(beta[i+1]) < tolerance:
                    alpha_1, eig_1 = self.implicit_qr_algorithm(alpha[:i+2], beta[:i+2],eigenvectors=eigenvectors[:,:i+2])
                    alpha_2, eig_2 = self.implicit_qr_algorithm(alpha[i+2:], beta[i+2:],eigenvectors=eigenvectors[:,i+2:])
                    return np.concatenate((alpha_1, alpha_2)), np.concatenate((eig_1, eig_2), axis=1)
                
                x = beta[i+1]

        return alpha, eigenvectors


