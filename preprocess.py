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
    def __init__(self, docs, name):
        """
        The class is initialized with the documents stored as a dictionary, 
        with docIDs as keys and the list of terms as values.
        """
        self.A = docs
        m, n = docs.shape
        self.left = m < n
        self.n = n
        self.name = name

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
                respone_time = 0
                for q_ind in range(num_of_query):
                    start_respone = time.time()
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
                    end_respone = time.time()
                    respone_time = respone_time + end_respone - start_respone
                    data_k[f'q{q_ind+1}'] = (np.argsort(scores)[::-1][:200]).tolist()
                
                data_k['av_respone'] = respone_time / num_of_query
                data[f'{k}'] = data_k

        with open(f'Output\{self.name}\lanczos_dc_{dc}.json', 'w') as f:
            json.dump(data, f)

    def sequential_lanczos(self, queries):
        data = {}
        for k in range(20, 301, 20):
            data_k = {}
            start_process = time.time()
            alpha, beta, q, norms = self.preprocess(k)
            end_process = time.time()
            data_k['process'] = end_process - start_process

            num_of_query = queries.shape[1]
            start_respone = time.time()
            for q_ind in range(num_of_query):
                scores = self.response(q, queries[:,q_ind])
                scores = scores/np.sqrt(norms)           
                data_k[f'q{q_ind+1}'] = (np.argsort(scores)[::-1][:200]).tolist()
            end_respone = time.time()
            data_k['av_respone'] = (end_respone - start_respone) / num_of_query
            data[f'{k}'] = data_k
        data[f'{k}'] = data_k
        with open(f'Output\{self.name}\lanczos_dc_1.json', 'w') as f:
            json.dump(data, f)

    def preprocess_lsi(self, k, indices=None):
        alpha, beta, q, norms = self.preprocess(k+1, indices=indices)
        eig_values, eig_vectors = self.implicit_qr_algorithm(alpha=alpha, beta=beta)
        eig_vectors = q.dot(eig_vectors)
        return eig_values[:-1], eig_vectors[:,:-1]

    def sequential_lsi(self, queries):
        data = {}
        for k in range(20, 301, 20):
            data_k = {}
            start_process = time.time()
            eig_values, eig_vectors = self.preprocess_lsi(k)
            end_process = time.time()
            data_k['process'] = end_process - start_process

            num_of_query = queries.shape[1]
            start_respone = time.time()
            for q_ind in range(num_of_query):
                if self.left:
                    scores = eig_vectors.T @ queries[:,q_ind]
                    scores = eig_vectors @ scores
                    scores = self.A.T @ scores
                    norms = np.linalg.norm(eig_vectors.T @ self.A, axis=0)
                    scores = scores[:,0] / norms
                else:
                    scores = self.A.T @ queries[:,0]
                    scores = eig_vectors.T @ scores
                    scores = eig_vectors @ scores 
                    norms = (eig_vectors * eig_vectors) @ eig_values
                    scores = scores[:,0] / np.sqrt(norms)
                          
                data_k[f'q{q_ind+1}'] = (np.argsort(scores)[::-1][:200]).tolist()
            end_respone = time.time()
            data_k['av_respone'] = (end_respone - start_respone) / num_of_query
            data[f'{k}'] = data_k
        data[f'{k}'] = data_k
        with open(f'Output\{self.name}\lsi_dc_1.json', 'w') as f:
            json.dump(data, f)

    def sequential_lsi_scipy(self, queries):
        data = {}
        for k in range(20, 301, 20):
            data_k = {}
            start_process = time.time()
            u, s, vt = svds(self.A, k)
            end_process = time.time()
            data_k['process'] = end_process - start_process

            num_of_query = queries.shape[1]
            start_respone = time.time()
            for q_ind in range(num_of_query):
                scores = u.T @ queries[:,q_ind]
                scores = s * scores
                scores = vt.T @ scores
                norms = (vt.T * vt.T) @ (s * s)
                scores = scores[:,0] / np.sqrt(norms)
                          
                data_k[f'q{q_ind+1}'] = (np.argsort(scores)[::-1][:200]).tolist()
            end_respone = time.time()
            data_k['av_respone'] = (end_respone - start_respone) / num_of_query
            data[f'{k}'] = data_k
        data[f'{k}'] = data_k
        with open(f'Output\{self.name}\sci_dc_1.json', 'w') as f:
            json.dump(data, f)

    def lsi_respone(self, args):
        eig_values, eig_vectors, query, indices = args
        if np.all(indices==None):
            A = self.A
        elif self.left:
            A = self.A[indices, :]
            query = query[indices]
        else:
            A = self.A[:, indices]
        
        if self.left:
            scores = eig_vectors.T @ query
            scores = eig_vectors @ scores
            scores = A.T @ scores
            norms = np.linalg.norm(eig_vectors.T @ A, axis=0)
            return scores[:,0], norms * norms
        else:
            scores = A.T @ query
            scores = eig_vectors.T @ scores
            scores = eig_vectors @ scores 
            norms = (eig_vectors * eig_vectors) @ eig_values
            return scores[:,0], norms 
        
    def parallel_lsi(self, queries, dc=2):
        parts = self.split_data(dc=dc)
        data = {}
        with ipp.Cluster(n=dc) as rc:
            # get a view on the cluster
            view = rc.load_balanced_view()
            for k in range(20, 301, 20):
                data_k = {}
                # submit the tasks
                start_process = time.time()
                asyncresult = view.map_async(lambda indices: self.preprocess_lsi(k, indices), parts)
                # wait interactively for results
                asyncresult.wait_interactive()
                
                # retrieve actual results
                values = []
                vectors = []
                if self.left:
                    norms = np.zeros(self.n)
                else:
                    norms = []
                for i, (eig_values, eig_vectors) in enumerate(asyncresult.get()):
                    values.append(eig_values)
                    vectors.append(eig_vectors)

                end_process = time.time()
                data_k['process'] = end_process - start_process
                
                num_of_query = queries.shape[1]
                start_respone = time.time()
                for q_ind in range(num_of_query):
                    respone_args = [(values[i], vectors[i], queries[:,q_ind], parts[i]) for i in range(dc)]
                    asyncresult = view.map_async(self.lsi_respone, respone_args)
                    asyncresult.wait_interactive()

                    if self.left:
                        scores = np.zeros(self.n)
                        norms = np.zeros(self.n)
                        for cos, norm in asyncresult.get():
                            scores = scores + cos
                            norms = norms + norm
                        scores = scores/np.sqrt(norms)
                    else:
                        scores = np.full(self.n, -1.0)
                        for i, (cos, norm) in enumerate(asyncresult.get()):
                            cos = cos / np.sqrt(norm)
                            for ind , j in enumerate(parts[i]):
                                if scores[j] < cos[ind]:
                                    scores[j] = cos[ind]
                                    
                    data_k[f'q{q_ind+1}'] = (np.argsort(scores)[::-1][:200]).tolist()
                end_respone = time.time()
                data_k['av_respone'] = (end_respone - start_respone) / num_of_query
                data[f'{k}'] = data_k

        with open(f'Output\{self.name}\lsi_dc_{dc}.json', 'w') as f:
            json.dump(data, f)
    
    def sci_preprocess(self, k, indices=None):
        if np.all(indices==None):
            A = self.A
        elif self.left:
            A = self.A[indices, :]
        else:
            A = self.A[:, indices]
        return svds(A, k=k)

    def sci_respone(self, args):
        u, s, vt, query, indices = args
        
        if self.left:
            query = query[indices]
        
        scores = (u.T @ query)[:,0]
        scores = s * scores
        scores = vt.T @ scores
        norms = (vt.T * vt.T) @ (s * s)
        return scores[::-1], norms[::-1]

    def parallel_sci(self, queries, dc=2):
        parts = self.split_data(dc=dc)
        data = {}
        with ipp.Cluster(n=dc) as rc:
            # get a view on the cluster
            view = rc.load_balanced_view()
            for k in range(20, 301, 20):
                data_k = {}
                # submit the tasks
                start_process = time.time()
                asyncresult = view.map_async(lambda indices: self.sci_preprocess(k, indices), parts)
                # wait interactively for results
                asyncresult.wait_interactive()
                
                # retrieve actual results
                us = []
                ss = []
                vts = []
                if self.left:
                    norms = np.zeros(self.n)
                else:
                    norms = []
                for i, (u, s, vt) in enumerate(asyncresult.get()):
                    us.append(u)
                    ss.append(s)
                    vts.append(vt)

                end_process = time.time()
                data_k['process'] = end_process - start_process
                
                num_of_query = queries.shape[1]
                start_respone = time.time()
                for q_ind in range(num_of_query):
                    respone_args = [(us[i], ss[i],vts[i], queries[:,q_ind], parts[i]) for i in range(dc)]
                    asyncresult = view.map_async(self.sci_respone, respone_args)
                    asyncresult.wait_interactive()

                    if self.left:
                        scores = np.zeros(self.n)
                        norms = np.zeros(self.n)
                        for cos, norm in asyncresult.get():
                            scores = scores + cos
                            norms = norms + norm
                        scores = scores/np.sqrt(norms)
                    else:
                        scores = np.full(self.n, -1.0)
                        for i, (cos, norm) in enumerate(asyncresult.get()):
                            cos = cos / np.sqrt(norm)
                            for ind , j in enumerate(parts[i]):
                                if scores[j] < cos[ind]:
                                    scores[j] = cos[ind]
                                    
                    data_k[f'q{q_ind+1}'] = (np.argsort(scores)[::-1][:200]).tolist()
                end_respone = time.time()
                data_k['av_respone'] = (end_respone - start_respone) / num_of_query
                data[f'{k}'] = data_k

        with open(f'Output\{self.name}\sci_dc_{dc}.json', 'w') as f:
            json.dump(data, f)
    
    def CompareToAb(self, query, left = False, reortho=True):
        logs = {}
        k = 100
        A = self.A
        m, n = A.shape
        ATb = A.T @ query
        if left:
            v = np.random.rand(m)
            q = np.zeros((m, k))
            s_hat = query
        else:
            v = np.random.rand(n)
            q = np.zeros((n, k))
            s_hat = ATb
        epsilon = np.sqrt(np.finfo(np.float64).eps)
        beta = 0
        alpha = 0
        q[:,0] = v/np.linalg.norm(v)
        u, sigma, vt = svds(A, 5)
        s = np.zeros(n)
        if left:
            s = np.zeros(m)
        for i in range(k-1):
            q_dot_query = q[:,i] @ s_hat
            s = s + q_dot_query * q[:,i]
            if left:
                x = A.T @  s
                logs[i] = (vt.T[:,4] @ ATb)[0] - (x @ vt.T[:,4]), (vt.T[:,3] @ ATb)[0] - (x @ vt.T[:,3]), (vt.T[:,2] @ ATb)[0] - (x @ vt.T[:,2])
                q_hat = A.T.dot(q[:,i])
                w = A.dot(q_hat) - beta * q[:,i-1]
            else:
                logs[i] = (vt.T[:,4] @ ATb)[0] - (s @ vt.T[:,4]), (vt.T[:,3] @ ATb)[0] - (s @ vt.T[:,3]), (vt.T[:,2] @ ATb)[0] - (s @ vt.T[:,2])
                q_hat = A.dot(q[:,i])
                w = A.T.dot(q_hat) - beta * q[:,i-1]

            alpha = w.dot(q[:,i])
            w = w - alpha * q[:,i]
            
            # start partial reorthonization
            if reortho:
                bound = np.linalg.norm(w) * epsilon
                for j in range(i):
                    w_dotq = w @ q[:,j]
                    if abs(w_dotq) > bound:
                        w = w - w_dotq * q[:,j]
            # end partial reorthonization

            beta = np.linalg.norm(w)
            if beta == 0:
                break
            q[:,i+1] = w / beta
        with open(f'Output\{self.name}\CompareToAb{"_reorthor" if reortho else ""}_{"left" if left else "right"}.json', 'w') as f:
            json.dump(logs, f)

    def CompareToAk(self, query, k):    
        alpha, beta, q, a = self.preprocess(k)
        lanc = self.response(q, query)

        u, sigma, vt = svds(self.A, k=k)
        u = u[:, ::-1]
        sigma = sigma[::-1]
        v = vt.T[:, ::-1]
        scores = (u.T @ query)[:,0]
        scores = sigma * scores
        scores = v @ scores
        log = {}
        for i in range(k):
            log[i] = v[:,i] @ (lanc - scores)
        
        with open(f'Output\{self.name}\CompareToAk{"_reorthor" if reortho else ""}_{"left" if left else "right"}.json', 'w') as f:
            json.dump(log, f)

    def precision_recall(self, relevance, retrieve):
        precision = {}
        recall = {}
        num_of_query = len(relevance)
        exact_true = 0
        full = {}
        for i in range(1, 200, 1):
            true_positive = 0
            exact_true = 0
            count = 0
            for j in range(num_of_query):
                _true_positive = len(set(relevance[j+1]).intersection(set(retrieve[f'q{j+1}'][:i])))
                _exact_true = len(relevance[j+1])
                if _true_positive < _exact_true:
                    count += 1
                    true_positive += _true_positive
                    exact_true += _exact_true
                elif _true_positive == _exact_true and j not in full:
                    full[j] = True
                    count += 1
                    true_positive += _true_positive
                    exact_true += _exact_true

            precision[i] = true_positive/(count * i)
            recall[i] = true_positive/exact_true

        return precision, recall

    def load_retrieval(self, path, k):
        with open(path, "r") as f:
           data = json.load(f)
           return data[f'{k}']
    
    def individual_precision(self, relevance, retrieve, qid):
        exact_true = len(relevance[qid])
        result = 0.0
        for i in range(exact_true):
            true_positive = len(set(relevance[qid]).intersection(set(retrieve[f'q{qid}'][:i+1])))
            result += true_positive/(i+1)
        return result/exact_true
    
    def mean_precision(self, path, relevance):
        num_of_query = len(relevance)
        result = 0.0
        for i in range(num_of_query):
            result += self.individual_precision(relevance, retrieve, i+1)
        return result/num_of_query
    
    
        