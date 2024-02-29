import numpy as np
from scipy.sparse import csr_matrix

def left_project_process_sparse(A, k, v):
    m, n = A.shape
    q = np.zeros((k, m))
    a = np.zeros(k)
    beta = np.zeros(k+1)
    beta[0] = 0
    q[0] = v / np.linalg.norm(v)
    A_T = A.T.tocsr()  # Transpose of A in CSR format
    for i in range(1, k):
        q_hat = A_T.dot(q[i-1])
        w = A.dot(q_hat) - beta[i] * q[i-1]
        alpha = w.dot(q[i-1])
        a[i] = a[i-1] + q_hat.dot(q_hat)
        w = w - alpha * q[i-1]
        beta[i+1] = np.linalg.norm(w)
        if beta[i+1] == 0:
            break
        for j in range(i):
            omega_jp1i = w.dot(q[j]) / q[j].dot(q[j])
            if omega_jp1i > np.sqrt(np.finfo(float).eps):
                w = w - w.dot(q[j]) * q[j]
            if omega_jp1i < eta:  # Assuming eta is defined
                break
        q[i] = w / beta[i+1]
    return q[:k], a[:k]




def calculate_epsilon_j(n, beta_2, beta_iplus1):
    # Generate Psi from N(0, 0.6)
    Psi = np.random.normal(0, 0.6)
    
    # Calculate epsilon_j
    epsilon_j = n * np.finfo(float).eps * (beta_2 / beta_iplus1) * Psi
    
    return epsilon_j

def calculate_nu_ij(beta_jplus1, beta_iplus1):
    # Generate Theta from N(0, 0.3)
    Theta_value = np.random.normal(0, 0.3)
    
    # Calculate nu_ij
    nu_ij = (beta_jplus1 + beta_iplus1) * np.finfo(float).eps * Theta_value
    
    return nu_ij

def calculate_omega(m, i, j, beta, alpha, memo):
    
    def helper(i, j):
        # Check if the value has already been computed
        if (i, j) in memo:
            return memo[(i, j)]
        if (j, i) in memo:
            return memo[(j, i)]
        
        # Implement calculation of omega_ij based on provided properties
        if j == 0 or i == 0:
            result = 0
        elif j == i:
            result = 1
        elif j == i - 1 or j - 1 == i:
            result = calculate_epsilon_j(beta[1], beta[i])
        else:
            omega_ijplus = helper(i-1, j+1)
            omega_ij = helper(i-1, j)
            omega_iminusj = helper(i-2, j)
            omega_ijminus = helper(i-1, j-1)
            result = (beta[j]*omega_ijplus + (alpha[j-1] - alpha[i-1])*omega_ij\
                + beta[j-1]*omega_ijminus - beta[i-1]*omega_iminusj) / beta[i] \
                + calculate_nu_ij(beta[i], beta[j])
        
        # Memoize the computed value
        memo[(i, j)] = result
        return result
    
    # Call the helper function with arguments i and j
    return helper(i, j)
