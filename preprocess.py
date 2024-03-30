import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds, eigsh
import math
import ipyparallel as ipp
import time
import pandas as pd
from ipyparallel import Client
import json

class VectorSpaceModel():
    def __init__(self, docs):
        """
        The class is initialized with the documents stored as a dictionary, 
        with docIDs as keys and the list of terms as values.
        """
        self.A = docs
        m, n = docs.shape
        self.left = m < n
        self.n = n

    def preprocess(self, k, indices = None):
        # reortho = {}
        if np.all(indices==None):
            A = self.A
        elif self.left:
            A = self.A[indices, :]
        else:
            A = self.A[:, indices]
            
        m, n = A.shape
        if m < n:
            v = np.random.rand(m)
            q = np.zeros((m, k))
        else:
            v = np.random.rand(n)
            q = np.zeros((n, k))
        epsilon = np.sqrt(np.finfo(np.float64).eps)
        a = np.zeros(n)
        beta = np.zeros(k)
        alpha = np.zeros(k)
        q[:,0] = v/np.linalg.norm(v)
        
        for i in range(k-1):
            if m < n:
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
                    # reortho[(i,j)] = w_dotq
                    w = w - w_dotq * q[:,j]
            # end partial reorthonization

            beta[i+1] = np.linalg.norm(w)
            if beta[i+1] == 0:
                break
            q[:,i+1] = w / beta[i+1]

        return alpha, beta, q, a

    
    def response(self, q, query, indices=None):
        if np.all(indices==None):
            A = self.A
        elif self.left:
            A = self.A[indices, :]
            query = query[indices]
        else:
            A = self.A[:, indices]

        len, k = q.shape

        if self.left:
            s_hat = query
        else:
            s_hat = A.T.dot(query)

        s = np.zeros(len)

        for i in range(k):
            q_dot_query = q[:,i] @ s_hat
            s = s + q_dot_query * q[:,i]

        if self.left:
            return A.T.dot(s)
        else:
            return s
    
    def tri_response(self, alpha, beta, q1, query, indices=None):
        if np.all(indices==None):
            A = self.A
        elif self.left:
            A = self.A[indices, :]
        else:
            A = self.A[:, indices]
        k = len(alpha)
        len = len(indices)
        s_hat = np.zeros(len)

        if self.left:
            s_hat = query
        else:
            s_hat = A.T.dot(query)

        s = np.zeros(len)

        q = np.zeros((self.len, self.k))
        q[:,0] = self.q1

        # range k-1 if tridiag
        for i in range(k-1):
            q_dot_query = q[:,i] @ s_hat
            s = s + q_dot_query * q[:,i]

            if self.left:
                Aq = self.A.T.dot(q[:,i])
                w = self.A.dot(Aq)
            else:
                Aq = self.A.dot(q[:,i])
                w = self.A.T.dot(Aq)

            w = w - self.beta[i] * q[:,i-1] - self.alpha[i] * q[:,i]
            for j in range(i):
                if (i, j) in self.reortho:
                    w = w - self.reortho[(i,j)] * q[:,j]
            q[:,i+1] = w / self.beta[i+1]
                
        q_dot_query = q[:,k-1] @ s_hat
        s = s + q_dot_query * q[:,k-1]

        if self.left:
            return A.T.dot(s)
        else:
            return s


    def implicit_qr_algorithm(self, alpha, beta, eigs=None, tolerance=1e-10):
        n = len(alpha)
        if n < 3:
            return alpha, eigs
        if np.all(eigs == None):
            eigs = np.eye(n)

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

                eigs[:, [i, i + 1]] = eigs[:, [i, i + 1]] @ np.array([[c, s], [-s, c]])
                
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
                    alpha_1, eig_1 = self.implicit_qr_algorithm(alpha[:i+2], beta[:i+2],eigs=eigs[:,:i+2])
                    alpha_2, eig_2 = self.implicit_qr_algorithm(alpha[i+2:], beta[i+2:],eigs=eigs[:,i+2:])
                    return np.concatenate((alpha_1, alpha_2)), np.concatenate((eig_1, eig_2), axis=1)
                
                x = beta[i+1]

        return alpha, eigs

    def bisec_PDDP(self, indices=None, iter = 4):
        if np.all(indices==None):
            A = self.A
        elif self.left:
            A = self.A[indices, :]
        else:
            A = self.A[:, indices]

        m, n = A.shape

        if m < n:
            d = np.sum(A, axis=0)/m
            q = np.zeros((m, iter))
            v = np.random.rand(m)
        else:
            d = np.sum(A, axis=1)/n
            q = np.zeros((n, iter))
            v = np.random.rand(n)
        
        q[:,0] = v/np.linalg.norm(v)
        d = np.array(d).flatten()
        
        # lanczos
        alpha = np.zeros(iter)
        beta = np.zeros(iter)
        
        for i in range(iter-1):
            if m < n:
                q_hat = A.T.dot(q[:,i]) - np.full(m, q[:,i] @ d)
                w = A.dot(q_hat) - beta[i] * q[:,i-1] - np.sum(q_hat) * d
            else:
                q_hat = A.dot(q[:,i]) - np.sum(q[:,i]) * d
                w = A.T.dot(q_hat) - beta[i] * q[:,i-1] - np.full(n, q_hat @ d)

            alpha[i] = w.dot(q[:,i])
            w = w - alpha[i] * q[:,i]

            # # start full reorthonization
            # for j in range(i):
            #     w_dotq = w @ q[:,j]
            #     w = w - w_dotq * q[:,j]

            beta[i+1] = np.linalg.norm(w)
            if beta[i+1] == 0:
                break
            q[:,i+1] = w / beta[i+1]
        # end lanczos

        eigvalues, eigvectors = self.implicit_qr_algorithm(alpha=alpha, beta=beta)
        principal = q.dot(eigvectors[:,0])

        if m < n:
            median = np.median(principal)
            left = np.where(principal <= median)[0]
            right = np.where(principal > median)[0]
        else:
            lo_bound = np.quantile(principal,.45)
            up_bound = np.quantile(principal,.55)
            left = np.where(principal < up_bound)[0]
            right = np.where(principal > lo_bound)[0]
        
        if np.all(indices==None):
            return left, right
        else:
            return indices[left], indices[right]
    
    def split_data(self, dc=2):
        left, right = self.bisec_PDDP()
        parts = [left, right]
        if dc == 4:
            x1, x2 = self.bisec_PDDP(left)
            x3, x4 = self.bisec_PDDP(right)
            parts = [x1, x2, x3, x4]
        return parts
    
    def respone_wrapper(self, args):
        q, query, indices = args
        return self.response(q, query, indices=indices)

    def parallel_lanczos(self, queries, dc=2):
        parts = self.split_data(dc=dc)
        data = {}
        with ipp.Cluster(n=dc) as rc:
            # get a view on the cluster
            view = rc.load_balanced_view()
            for k in range(20, 301, 20):
                data_k = {}
                # submit the tasks
                start_process = time.time()
                asyncresult = view.map_async(lambda indices: self.preprocess(k, indices), parts)
                # wait interactively for results
                asyncresult.wait_interactive()
                
                # retrieve actual results
                lanczos = []
                if self.left:
                    norms = np.zeros(self.n)
                else:
                    norms = []
                for i, (alpha, beta, q, n) in enumerate(asyncresult.get()):
                    lanczos.append(q)
                    if self.left:
                        norms = norms + n
                    else:
                        norms.append(n)
                end_process = time.time()
                data_k['process'] = end_process - start_process
                
                num_of_query = queries.shape[1]
                start_respone = time.time()
                for q_ind in range(num_of_query):
                    respone_args = [(lanczos[i], queries[:,q_ind], parts[i]) for i in range(dc)]
                    asyncresult = view.map_async(self.respone_wrapper, respone_args)
                    asyncresult.wait_interactive()
                    coses = asyncresult.get()

                    if self.left:
                        scores = np.zeros(self.n)
                        for cos in coses:
                            scores = scores + cos
                        scores = scores/np.sqrt(norms)
                    else:
                        scores = np.full(self.n, -1.0)
                        for i in range(dc):
                            cos = coses[i] / np.sqrt(norms[i])
                            for ind , j in enumerate(parts[i]):
                                if scores[j] < cos[ind]:
                                    scores[j] = cos[ind]
                                    
                    data_k[f'q{q_ind+1}'] = scores.tolist()
                end_respone = time.time()
                data_k['av_respone'] = (end_respone - start_respone) / num_of_query
                data[f'{k}'] = data_k

        with open(f'Output\lanczos_dc_{dc}.json', 'w') as f:
            json.dump(data, f)