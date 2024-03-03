import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

class VectorSpaceModel():
    def __init__(self, docs):
        """
        The class is initialized with the documents stored as a dictionary, 
        with docIDs as keys and the list of terms as values.
        """
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.A = self.tfidf.fit_transform(docs)

    def convert_docs_tdidf(self):
        pass

#     def preprocess(self):
#         A = self.A
#         k = self.k
#         len = self.len
#         n = self.n
#         eta = epsilon ** (3/4)
#         q = np.zeros((k, len))
#         a = np.zeros(n)
#         beta = np.zeros(k)
#         alpha = np.zeros(k)
#         v = np.random.rand(len)
#         q[0] = v / np.linalg.norm(v)
#         A_T = A.T.tocsr()  # Transpose of A in CSR format
#         memo = {}
#         for i in range(k):
#             if self.left:
#                 q_hat = A_T.dot(q[i])
#                 w = A.dot(q_hat) - beta[i] * q[i-1]
#                 alpha[i] = w.dot(q[i])
#                 a = a + np.square(q_hat)
#             else:
#                 q_hat = A.dot(q[i])
#                 w = A_T.dot(q_hat) - beta[i] * q[i-1]
#                 alpha[i] = w.dot(q[i])
#                 a = a + alpha[i]*np.square(q[i]) + 2*beta[i]*np.multiply(q[i], q[i-1])

#             w = w - alpha[i] * q[i]
#             beta[i+1] = np.linalg.norm(w)
#             if beta[i+1] == 0:
#                 break
#             for j in range(i):
#                 omega = calculate_omega(len, i, j, beta, alpha, memo)
#                 if omega > np.sqrt(np.finfo(float).eps):
#                     w = w - w.dot(q[j]) * q[j]
#                 if omega < eta:
#                     break
#             q[i+1] = w / beta[i+1]
#         self.norms = a
#         if self.tridiag:
#             self.alpha = alpha
#             self.beta = beta
#             self.q1 = q[0]
#         else:
#             self.lanczos_vectors = q

# # Example usage:
# # A_sparse = csr_matrix(A)  # Convert A to CSR format
# # k = 5  # Number of vectors to compute
# # q, a = left_project_process_sparse(A_sparse, k)
# # print(q, a)


#     def _calculate_epsilon_j(self, n, beta_2, beta_iplus1):
#         # Generate Psi from N(0, 0.6)
#         Psi = np.random.normal(0, 0.6)
        
#         # Calculate epsilon_j
#         epsilon_j = n * np.finfo(float).eps * (beta_2 / beta_iplus1) * Psi
        
#         return epsilon_j

#     def _calculate_nu_ij(self, beta_jplus1, beta_iplus1):
#         # Generate Theta from N(0, 0.3)
#         Theta_value = np.random.normal(0, 0.3)
        
#         # Calculate nu_ij
#         nu_ij = (beta_jplus1 + beta_iplus1) * np.finfo(float).eps * Theta_value
        
#         return nu_ij

#     def _calculate_omega(self, m, i, j, beta, alpha, memo):
#         def helper(i, j):
#             # Check if the value has already been computed
#             if (i, j) in memo:
#                 return memo[(i, j)]
#             if (j, i) in memo:
#                 return memo[(j, i)]
            
#             # Implement calculation of omega_ij based on provided properties
#             if j < 0 or i < 0:
#                 result = 0
#             elif j == i:
#                 result = 1
#             elif j == i - 1 or j - 1 == i:
#                 result = _calculate_epsilon_j(m, beta[1], beta[i+1])
#             else:
#                 omega_ijplus = helper(i, j+1)
#                 omega_ij = helper(i, j)
#                 omega_iminusj = helper(i-1, j)
#                 omega_ijminus = helper(i, j-1)
#                 result = (beta[j+1]*omega_ijplus + (alpha[j] - alpha[i])*omega_ij\
#                     + beta[j]*omega_ijminus - beta[i]*omega_iminusj) / beta[i+1] \
#                     + _calculate_nu_ij(beta[i+1], beta[j+1])
            
#             # Memoize the computed value
#             memo[(i, j)] = result
#             return result
        
#         # Call the helper function with arguments i and j
#         return helper(i, j)


#     def response(self, query):
#         A_T = A.T.tocsr()
#         if not self.left:
#             query = A_T.dot(query)    
#         s = np.zeros(self.n)  # Initialize s_hat with zeros
#         q = self.q1
#         if self.tridiag:
#             q0 = np.zeros_like(q) 
#         for i in range(self.k):
#             s += np.dot(q, query) * q  # Update s_hat
#             if self.tridiag:
#                 if self.beta[i+1] == 0:
#                     break
#                 Aq= A.dot(q)
#                 w = A_T.dot(Aq) - self.beta[i]*q0 - self.alpha[i]*q
#                 q0 = q
#                 q = w/self.beta[i+1]
#             else:
#                 q = self.lanczos_vectors[i+1]
#         if self.left:
#             self.scores = A_T.dot(s)
#         else:
#             self.scores = s