import numpy as np

# Flatten an NxN covariance matrix to a vector of size N(N+1)/2 containing upper triangular elements
def flatten_cov(M):
    idx = np.triu_indices_from(M)
    return M[idx]         # shape: (p,), where p = N(N+1)/2

# Unflatten a vector of size N(N+1)/2 containing upper triangular elements to an NxN covariance matrix
def unflatten_cov(v, n):
    M = np.zeros((n, n))
    idx = np.triu_indices(n)
    M[idx] = v # fill in upper triangular
    M[(idx[1], idx[0])] = v # fill in lower triangular
    return M

# Enforce positive semi-definiteness on a matrix
def enforce_psd(M, eps=1e-6):
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals_clipped = np.clip(eigvals, eps, None)
    M_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return M_psd

# Compute Frobenius norm error between two matrices
def frobenius_error(M1, M2):
    return np.linalg.norm(M1.flatten() - M2.flatten())