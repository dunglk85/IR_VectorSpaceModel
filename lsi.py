import re
import numpy as np
from collections import defaultdict, OrderedDict, Counter
from math import log, sqrt
from os import listdir
from os.path import isfile, join

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class VectorSpaceModel():
    def __init__(self, docs):
        """
        The class is initialized with the documents stored as a dictionary, 
        with docIDs as keys and the list of terms as values.
        """
        self.docs = []
        for i, d in enumerate(docs):
            self.docs.append(self.preprocess_tokens(docs[i]))
        self.n_docs = len(self.docs)
        self.index = self.inverted_index()
        self.vocab = self.get_vocabulary()
        self.n_terms = len(self.vocab)
        self.tfidf = None


def lanpro(A, nin, kmax, r, options, Q_k, T_k, anorm):
    if A is None or nin is None or kmax is None or r is None or options is None or Q_k is None or T_k is None or anorm is None:
        raise ValueError('Not enough input arguments.')
    
    if isinstance(A, np.ndarray) or isinstance(A, dict):
        if isinstance(A, np.ndarray):
            m, n = A.shape
            if m != n or not np.allclose(A, A.T) or not np.isreal(A).all():
                raise ValueError('A must be real symmetric')
        elif isinstance(A, dict):
            m, n = A['L'].shape
        if T_k is None or T_k.size == 0:
            anorm = None
            est_anorm = True
        else:
            est_anorm = False
        if Q_k is None:
            Q_k = np.zeros((n, kmax))
        if T_k is None:
            T_k = np.zeros((kmax, kmax))
        if r is None or r.size == 0:
            r = np.random.rand(n) - 0.5
        if kmax is None or kmax.size == 0:
            kmax = max(10, n / 10)
    else:
        if nin is None:
            raise ValueError('Not enough input arguments.')
        if not callable(A):
            raise ValueError('A must be a function.')
        n = nin
        if anorm is None or anorm.size == 0:
            anorm = None
            est_anorm = True
        else:
            est_anorm = False
        if Q_k is None:
            Q_k = np.zeros((n, kmax))
        if T_k is None:
            T_k = np.zeros((kmax, kmax))
        if r is None or r.size == 0:
            r = np.random.rand(n) - 0.5
        if kmax is None or kmax.size == 0:
            kmax = max(10, n / 10)
    
    delta = np.sqrt(np.finfo(float).eps / kmax)
    eta = np.power(np.finfo(float).eps, 3 / 4) / np.sqrt(kmax)
    cgs = 0
    elr = 1
    deflate = 0
    
    if options is not None and isinstance(options, dict):
        if 'delta' in options:
            delta = options['delta']
        if 'eta' in options:
            eta = options['eta']
        if 'cgs' in options:
            cgs = options['cgs']
        if 'elr' in options:
            elr = options['elr']
        if 'Y' in options:
            deflate = len(options['Y']) > 0
    
    np = 0
    nr = 0
    ierr = 0
    eps1 = np.sqrt(n) * np.finfo(float).eps / 2
    gamma = 1 / np.sqrt(2)
    
    if Q_k is None or Q_k.size == 0:
        alpha = np.zeros(kmax + 1)
        beta = np.zeros(kmax + 1)
        Q_k = np.zeros((n, kmax))
        q = np.zeros(n)
        beta[0] = np.linalg.norm(r)
        omega = np.zeros(kmax)
        omega_max = np.zeros(kmax)
        omega_old = np.zeros(kmax)
        omega[0] = 0
        force_reorth = 0
        j0 = 1
    else:
        j = Q_k.shape[1]
        Q_k = np.hstack((Q_k, np.zeros((n, kmax - j))))
        alpha = np.zeros(kmax + 1)
        beta = np.zeros(kmax + 1)
        alpha[:j] = np.diag(T_k)
        if j > 1:
            beta[1:j] = np.diag(T_k, -1)
        q = Q_k[:, j - 1]
        beta[j] = np.linalg.norm(r)
        if j < kmax and beta[j] * delta < anorm * eps1:
            fro = 1
        if np.isfinite(delta):
            int = np.arange(j)
            r, beta[j], rr = reorth(Q_k, r, beta[j], int, gamma, cgs)
            np += rr * j
            nr = 1
            force_reorth = 1
        else:
            force_reorth = 0
        anorm = np.sqrt(np.linalg.norm(T_k.T @ T_k, 1))
        omega = eps1 * np.ones(kmax)
        omega_max = omega
        omega_old = omega
        j0 = j + 1
    
    if delta == 0:
        fro = 1
    else:
        fro = 0
    
    for j in range(j0, kmax):
        q_old = q
        if beta[j] == 0:
            q = r
        else:
            q = r / beta[j]
        Q_k[:, j] = q
        if callable(A):
            u = A(q)
        else:
            u = A @ q
        r = u - beta[j] * q_old
        alpha[j] = q @ r
        r = r - alpha[j] * q
        beta[j + 1] = np.linalg.norm(r)
        if beta[j + 1] < gamma * beta[j] and elr:
            if j == 0:
                t1 = 0
                for i in range(2):
                    t = q @ r
                    r = r - q * t
                    t1 += t
                alpha[j] += t1
            elif j > 0:
                t1 = q_old @ r
                t2 = q @ r
                r = r - (q_old * t1 + q * t2)
                if beta[j] != 0:
                    beta[j] += t1
                alpha[j] += t2
            beta[j + 1] = np.linalg.norm(r)
        if est_anorm and beta[j + 1] != 0:
            anorm = update_gbound(anorm, alpha, beta, j)
        if j > 0 and not fro and beta[j + 1] != 0:
            omega, omega_old = update_omega(omega, omega_old, j, alpha, beta, eps1, anorm)
            omega_max[j] = np.max(np.abs(omega))
        if j > 0 and (fro or force_reorth or omega_max[j] > delta) and beta[j + 1] != 0:
            if fro:
                int = np.arange(j)
            else:
                if force_reorth == 0:
                    force_reorth = 1
                    int = compute_int(omega, j, delta, eta, 0, 0, 0)
                else:
                    force_reorth = 0
            r, beta[j + 1], rr = reorth(Q_k, r, beta[j + 1], int, gamma, cgs)
            omega[int] = eps1
            np += rr * len(int)
            nr += 1
        else:
            beta[j + 1] = np.linalg.norm(r)
        if deflate:
            r, beta[j + 1], rr = reorth(options['Y'], r, beta[j + 1], np.arange(options['Y'].shape[1]), gamma, cgs)
        if j < kmax and beta[j + 1] < n * anorm * np.finfo(float).eps:
            beta[j + 1] = 0
            bailout = 1
            for attempt in range(3):
                r = np.random.rand(n) - 0.5
                if callable(A):
                    r = A(r)
                else:
                    r = A @ r
                nrm = np.linalg.norm(r)
                int = np.arange(j)
                r, nrmnew, rr = reorth(Q_k, r, nrm, int, gamma, cgs)
                omega[int] = eps1
                np += rr * len(int)
                nr += 1
                if nrmnew > 0:
                    bailout = 0
                    break
            if bailout:
                ierr = -j
                break
            else:
                r = r / nrmnew
                force_reorth = 1
                if delta > 0:
                    fro = 0
        elif j < kmax and not fro and beta[j + 1] * delta < anorm * eps1:
            fro = 1
            ierr = j
    
    T_k = spdiags(np.vstack((beta[1:j + 1], alpha[:j + 1], beta[:j + 1])).T, [-1, 0, 1], j, j)
    
    if T_k is None:
        Q_k = T_k
    elif j != Q_k.shape[1]:
        Q_k = Q_k[:, :j]
    
    work = np.array([nr, np])
    
    return Q_k, T_k, r, anorm, ierr, work


def update_omega(omega, omega_old, j, alpha, beta, eps1, anorm):
    T = eps1 * anorm
    binv = 1 / beta[j + 1]
    omega_old = omega.copy()
    # Update omega(1) using omega(0)==0.
    omega_old[0] = beta[1] * omega[1] + (alpha[0] - alpha[j]) * omega[0] - beta[j] * omega_old[0]
    omega_old[0] = binv * (omega_old[0] + np.sign(omega_old[0]) * T)
    # Update remaining components.
    for k in range(1, j - 1):
        omega_old[k] = beta[k + 1] * omega[k + 1] + (alpha[k] - alpha[j]) * omega[k] + beta[k] * omega[k - 1] - beta[j] * omega_old[k]
        omega_old[k] = binv * (omega_old[k] + np.sign(omega_old[k]) * T)
    omega_old[j - 2] = binv * T
    # Swap omega and omega_old.
    temp = omega.copy()
    omega = omega_old
    omega_old = temp
    omega[j] = eps1
    return omega, omega_old



def update_gbound(anorm, alpha, beta, j):
    if j == 1:
        i = j
        scale = max(abs(alpha[i]), abs(beta[i+1]))
        alpha[i] = alpha[i] / scale
        beta[i+1] = beta[i+1] / scale
        anorm = 1.01 * scale * np.sqrt(alpha[i]**2 + beta[i+1]**2 + abs(alpha[i] * beta[i+1]))
    elif j == 2:
        i = 1
        scale = max(max(abs(alpha[0:2]), max(abs(beta[1:3]))))
        alpha[0:2] = alpha[0:2] / scale
        beta[1:3] = beta[1:3] / scale
        anorm = max(anorm, scale * np.sqrt(alpha[i]**2 + beta[i+1]**2 +
                                           abs(alpha[i] * beta[i+1] + alpha[i+1] * beta[i+1]) +
                                           abs(beta[i+1] * beta[i+2])))
        i = 2
        anorm = max(anorm, scale * np.sqrt(abs(beta[i] * alpha[i-1] + alpha[i] * beta[i]) +
                                           beta[i]**2 + alpha[i]**2 + beta[i+1]**2 +
                                           abs(alpha[i] * beta[i+1])))
    elif j == 3:
        scale = max(max(abs(alpha[0:3]), max(abs(beta[1:4]))))
        alpha[0:3] = alpha[0:3] / scale
        beta[1:4] = beta[1:4] / scale
        i = 2
        anorm = max(anorm, scale * np.sqrt(abs(beta[i] * alpha[i-1] + alpha[i] * beta[i]) +
                                           beta[i]**2 + alpha[i]**2 + beta[i+1]**2 +
                                           abs(alpha[i] * beta[i+1] + alpha[i+1] * beta[i+1]) +
                                           abs(beta[i+1] * beta[i+2])))
        i = 3
        anorm = max(anorm, scale * np.sqrt(abs(beta[i] * beta[i-1]) +
                                           abs(beta[i] * alpha[i-1] + alpha[i] * beta[i]) +
                                           beta[i]**2 + alpha[i]**2 + beta[i+1]**2 +
                                           abs(alpha[i] * beta[i+1])))
    else:
        i = j-1
        anorm1 = np.sqrt(abs(beta[i] * beta[i-1]) +
                         abs(beta[i] * alpha[i-1] + alpha[i] * beta[i]) +
                         beta[i]**2 + alpha[i]**2 + beta[i+1]**2 +
                         abs(alpha[i] * beta[i+1] + alpha[i+1] * beta[i+1]) +
                         abs(beta[i+1] * beta[i+2]))
        if np.isfinite(anorm1):
            anorm = max(anorm, anorm1)
        i = j
        anorm1 = np.sqrt(abs(beta[i] * beta[i-1]) +
                         abs(beta[i] * alpha[i-1] + alpha[i] * beta[i]) +
                         beta[i]**2 + alpha[i]**2 + beta[i+1]**2 +
                         abs(alpha[i] * beta[i+1]))
        if np.isfinite(anorm1):
            anorm = max(anorm, anorm1)
    return anorm
