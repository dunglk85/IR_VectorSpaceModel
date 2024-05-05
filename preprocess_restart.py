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

class VectorSpaceModel2():
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
            Q = np.zeros((m, k+1))
        else:
            v = np.random.rand(n)
            Q = np.zeros((n, k+1))
        epsilon = np.sqrt(np.finfo(np.float64).eps)
        a = np.zeros(n)
        beta = 0.0
        
        Q[:,0] = v/np.linalg.norm(v)
        
        for i in range(k):
            if m < n:
                q_hat = A.T.dot(Q[:,i])
                w = A.dot(q_hat) - beta * Q[:,i-1]
                a = a + np.square(q_hat)
            else:
                q_hat = A.dot(Q[:,i])
                w = A.T.dot(q_hat) - beta * Q[:,i-1]
                a = a + (q_hat @ q_hat)* np.square(Q[:,i])
            
            alpha = w.dot(Q[:,i])
            w = w - alpha * Q[:,i]

            # start partial reorthonization
            bound = np.linalg.norm(w) * epsilon
            for j in range(i):
                w_dotq = w @ Q[:,j]
                if abs(w_dotq) > bound:
                    w = w - w_dotq * Q[:,j]
            # end partial reorthonization

            beta = np.linalg.norm(w)
            if beta == 0:
                break
            Q[:,i+1] = w / beta

        return Q, a

    def response(self, Q, query, indices=None):
        if np.all(indices==None):
            A = self.A
        elif self.left:
            A = self.A[indices, :]
            query = query[indices]
        else:
            A = self.A[:, indices]

        if self.left:
            s = Q.T @ query
            s = Q @ s
            s = A.T @ s
            return s.squeeze()
        else:
            s = A.T @ query
            s = Q.T @ s
            s = Q @ s
            return s.squeeze()
    
    def bisec_PDDP(self, indices=None):
        if np.all(indices==None):
            A = self.A
        elif self.left:
            A = self.A[indices, :]
        else:
            A = self.A[:, indices]
        
        m, n = A.shape
        u, s, vt = svds(A, k=1)
        s1 = s[0]

        if m < n:
            u1= u[:,0]
            d = np.sum(A, axis=0)
            d = np.array(d)
            d = d[:,0]/(s1*s1*math.sqrt(m))
            principal = u1 - d

            median = np.median(principal)
            left = np.where(principal <= median)[0]
            right = np.where(principal > median)[0]
        else:
            v1= vt.T[:,0]
            d = np.sum(A, axis=1)
            d = np.array(d)
            d = A.T @ d
            d = d[:,0]/(s1*s1*math.sqrt(n))
            principal = v1 - d

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

    def sequential_lanczos(self, queries):
        data = {}
        for k in range(50, 801, 50):
            data_k = {}
            start_process = time.time()
            q, norms = self.preprocess(k)
            end_process = time.time()
            data_k['process'] = end_process - start_process

            num_of_query = queries.shape[1]
            respone_time = 0.0
            for q_ind in range(num_of_query):
                start_respone = time.time()
                scores = self.response(q, queries[:,q_ind])
                scores = scores/np.sqrt(norms)           
                data_k[f'q{q_ind+1}'] = (np.argsort(scores)[::-1][:200]).tolist()
                end_respone = time.time()
                respone_time = respone_time + end_respone - start_respone
            data_k['av_respone'] = respone_time / num_of_query
            data[f'{k}'] = data_k
        data[f'{k}'] = data_k
        with open(f'Output\{self.name}\lanczos_dc_1.json', 'w') as f:
            json.dump(data, f)

    def parallel_lanczos(self, queries, dc=2):
        start_subdividing = time.time()
        parts = self.split_data(dc=dc)
        end_subdividing = time.time()
        data = {}
        with ipp.Cluster(n=dc) as rc:
            # get a view on the cluster
            view = rc.load_balanced_view()
            for k in range(0, 301, 20):
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
                for i, (Q, n) in enumerate(asyncresult.get()):
                    lanczos.append(Q)
                    if self.left:
                        norms = norms + n
                    else:
                        norms.append(n)
                end_process = time.time()
                data_k['process'] = end_process - start_process + end_subdividing - start_subdividing
                
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
        
        scores = u.T @ query
        scores = s * scores[:,0]
        scores = vt.T @ scores
        norms = (vt.T * vt.T) @ (s * s)
        return scores, norms

    def sequential_sci(self, queries):
        data = {}
        for k in range(50, 801, 50):
            data_k = {}
            start_process = time.time()
            u, s, vt = svds(self.A, k)
            end_process = time.time()
            data_k['process'] = end_process - start_process

            num_of_query = queries.shape[1]
            
            respone_time = 0.0
            for q_ind in range(num_of_query):
                start_respone = time.time()
                scores = u.T @ queries[:,q_ind]
                scores = s * scores[:,0]
                scores = vt.T @ scores
                norms = (vt * vt).T @ (s * s)
                scores = scores/ np.sqrt(norms)
                end_respone = time.time()
                respone_time = respone_time + end_respone - start_respone
                indices = (np.argsort(scores)[::-1][:200]).tolist()      
                data_k[f'q{q_ind+1}'] = indices
            
            data_k['av_respone'] = respone_time / num_of_query
            data[f'{k}'] = data_k
        data[f'{k}'] = data_k
        with open(f'Output\{self.name}\sci_dc_1.json', 'w') as f:
            json.dump(data, f)

    def parallel_sci(self, queries, dc=2):
        start_subdividing = time.time()
        parts = self.split_data(dc=dc)
        end_subdividing = time.time()
        data = {}
        with ipp.Cluster(n=dc) as rc:
            # get a view on the cluster
            view = rc.load_balanced_view()
            for k in range(50, 801, 50):
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
                data_k['process'] = end_process - start_process + end_subdividing - start_subdividing
                
                num_of_query = queries.shape[1]
                respone_time = 0.0
                for q_ind in range(num_of_query):
                    start_respone = time.time()
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
                    end_respone = time.time()
                    respone_time = respone_time + end_respone - start_respone
                    scores = scores[::-1]               
                    data_k[f'q{q_ind+1}'] = (np.argsort(scores)[::-1][:200]).tolist()
                
                data_k['av_respone'] = respone_time / num_of_query
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
            Q = np.zeros((m, k))
            s_hat = query
        else:
            v = np.random.rand(n)
            Q = np.zeros((n, k))
            s_hat = ATb
        epsilon = np.sqrt(np.finfo(np.float64).eps)
        beta = 0
        alpha = 0
        Q[:,0] = v/np.linalg.norm(v)
        u, sigma, vt = svds(A, 5)
        s = np.zeros(n)
        if left:
            s = np.zeros(m)
        for i in range(k-1):
            q_dot_query = Q[:,i] @ s_hat
            s = s + q_dot_query * Q[:,i]
            if left:
                x = A.T @  s
                logs[i] = (vt.T[:,4] @ ATb)[0] - (x @ vt.T[:,4]), (vt.T[:,3] @ ATb)[0] - (x @ vt.T[:,3]), (vt.T[:,2] @ ATb)[0] - (x @ vt.T[:,2])
                q_hat = A.T.dot(Q[:,i])
                w = A.dot(q_hat) - beta * Q[:,i-1]
            else:
                logs[i] = (vt.T[:,4] @ ATb)[0] - (s @ vt.T[:,4]), (vt.T[:,3] @ ATb)[0] - (s @ vt.T[:,3]), (vt.T[:,2] @ ATb)[0] - (s @ vt.T[:,2])
                q_hat = A.dot(Q[:,i])
                w = A.T.dot(q_hat) - beta * Q[:,i-1]

            alpha = w.dot(Q[:,i])
            w = w - alpha * Q[:,i]
            
            # start partial reorthonization
            if reortho:
                bound = np.linalg.norm(w) * epsilon
                for j in range(i):
                    w_dotq = w @ Q[:,j]
                    if abs(w_dotq) > bound:
                        w = w - w_dotq * Q[:,j]
            # end partial reorthonization

            beta = np.linalg.norm(w)
            if beta == 0:
                break
            Q[:,i+1] = w / beta
        with open(f'Output\{self.name}\CompareToAb{"_reorthor" if reortho else ""}_{"left" if left else "right"}.json', 'w') as f:
            json.dump(logs, f)

    def CompareToAk(self, query, k):    
        q, a = self.preprocess(k)
        lanc = self.response(q, query)

        u, sigma, vt = svds(self.A, k=k)
        scores = (u.T @ query)
        scores = sigma * scores[:,0]
        scores = vt.T @ scores
        log = {}
        for i in range(k):
            log[i] = vt[k-i-1,:] @ (lanc - scores)
        with open(f'Output\{self.name}\CompareToAk.json', 'w') as f:
            json.dump(log, f)

    def precision_recall(self, relevance, retrieve):
        precision = []
        recall = []
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

            precision.append(true_positive/(count * i))
            recall.append(true_positive/exact_true)

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
    
    def mean_precision(self, relevance, retrieve):
        num_of_query = len(relevance)
        result = 0.0
        for i in range(num_of_query):
            result += self.individual_precision(relevance, retrieve, i+1)
        return result/num_of_query
    
        