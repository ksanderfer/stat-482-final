import numpy as np
import scipy.stats as sp
import scikitlearn as sk


# Use correlated GBM to simulate asset returns
def simulate_prices(mu, Sigma, times, seed=None):
       
    n_assets = len(mu)
    n_times = len(times)
    
    # Cholesky decomposition of covariance matrix
    L = np.linalg.cholesky(Sigma)
    
    # Initialize log returns matrix
    log_returns = np.zeros((n_times - 1, n_assets))
    
    # Loop over time steps
    for i in range(1, n_times):
        dt = t[i] - t[i-1]
        Z = np.random.normal(size=n_assets)          # independent normals
        dW = L @ Z * np.sqrt(dt)                     # correlated Brownian increments
        log_returns[i-1] = (mu - 0.5*np.diag(Sigma)) * dt + dW
        
    return log_returns

# Generate our testing/training "true" cov matrices
def generate_cov_matrices(n, num_matrices=10000, seed=67):
    matrices = []
    for i in range(num_matrices):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(n, n))
        Sig = A @ A.T
        D = np.sqrt(np.diag(Sig))
        corr = Sig / np.outer(D,D)
        matrices.append(corr)
    return matrices

#Randomize our matrices into "random" cov matrices using Wishart
#mild noise should be  noise factor of 10
def wishartize(mat, noise_factor):
    df = mat.shape()[0] + noise_factor
    return (mat, sp.wishart.rvs(df, mat/df))

# Construct our list of tuples of matrices: (true mat, random mat)
def construct_data(noise_factor, n, num_matrices=10000, seed=67):
    simulated = []
    true_mats = generate_cov_matrices(n, num_matrices, seed)
    for mat in true_mats:
        simulated.append((mat, wishartize(mat, noise_factor)))
    return true_mats, simulated

#takes in noisy wishart matrix, uses GBM to generate asset prices over a year. 
#mu is randomly chosen in a reasonable way
def generate_returns(mat):
    n = np.shape(mat)[0]
    mu = np.random.normal(0.1, 0.01, n) #this will put mus between 0.08 and 0.12 mostly
    times = np.linspace(1/252, 1, 252)
    price_mat = simulate_prices(mu, mat, times, seed=None)

#generates true matrices, simulated Wishart matrices, does GBM, and gets corresponding sample covariance matrices
def train_model():
    (true_mats, simulated) = construct_data(10, n = 10, num_matrices = 10000, seed = 67)
    sample_mats = []
    for simul_mat in simulated:
        sample_mats.append(np.cov(generate_returns(mat), rowvar=False))
    #now each data point is (true_mat, sample_mat)
    #once we do a linear regression on this we're done