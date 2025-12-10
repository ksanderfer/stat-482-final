import numpy as np
from generate_data import construct_dataset
from sklearn.covariance import LedoitWolf

# Get sample covariance matrix from returns
def sample_covariance (returns):
   return np.cov(returns, rowvar=False)

# Apply Ledoit-Wolf shrinkage to covariance matrix
def ledoit_wolf_shrinkage (returns):
    return LedoitWolf().fit(sample_covariance(returns)).covariance_

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

# Builds dataset for training regression models for covariance estimation
def build_training_dataset(num_matrices=1000, n=10, T=252, seed=67):
    dataset = construct_dataset(noise_factor=10, n=n, num_matrices=num_matrices, seed=seed)
    X = [r.flatten() for _, _, r in dataset]
    y = [flatten_cov(c) for c, _, _ in dataset]
    return np.array(X), np.array(y)