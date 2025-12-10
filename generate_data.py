import numpy as np
import scipy.stats as sp

# Generate "true" covariance matrices
def generate_cov_matrices (n, num_matrices=10000, seed=67):
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(num_matrices):
        A = rng.normal(size=(n, n))
        Sig = A @ A.T
        D   = np.sqrt(np.diag(Sig))
        corr = Sig / np.outer(D, D)
        stds = rng.uniform(0.1, 0.3, size=n)
        cov  = corr * np.outer(stds, stds)
        matrices.append(cov)
    return matrices

# Randomize a matrix using Wishart mild noise
def wishartize(mat, noise_factor = 10):
    df = mat.shape[0] + noise_factor
    return sp.wishart.rvs(df=df, scale=mat/df)

# Use correlated GBM to simulate asset returns
def simulate_returns(mu, Sigma, times, seed=None):
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    n_times = len(times)

    L = np.linalg.cholesky(Sigma)
    log_returns = np.zeros((n_times - 1, n_assets))

    for i in range(1, n_times):
        dt = times[i] - times[i-1]
        Z = rng.normal(size=n_assets)
        dW = L @ Z * np.sqrt(dt)
        log_returns[i-1] = (mu - 0.5 * np.diag(Sigma)) * dt + dW

    return log_returns

# Uses GBM to generate asset prices from a covariance matrix over a one year window. 
def generate_returns(Sigma):
    n = np.shape(Sigma)[0]
    mu = np.random.normal(0.1, 0.01, n) #this will put mus between 0.08 and 0.12 mostly
    times = np.linspace(0, 1, 253)
    price_mat = simulate_returns(mu, Sigma, times, seed=None)
    return price_mat

# Construct a dataset of true covariance matrices, their noisy Wishart versions, and simulated returns
def construct_dataset (noise_factor, n, num_matrices=10000, seed=67):
    simulated = []
    true_mats = generate_cov_matrices(n, num_matrices, seed)
    for mat in true_mats:
        noisy = wishartize(mat, noise_factor)
        returns = generate_returns(noisy)
        simulated.append((mat, np.linalg.cholesky(mat), returns))   # (true_cov, cholesky, returns)
    return simulated