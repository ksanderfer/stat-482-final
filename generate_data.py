import numpy as np
import scipy.stats as sp

# Use correlated GBM to simulate asset returns
def simulate_prices(mu, Sigma, times, seed=None):
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


# Generate our testing/training "true" cov matrices
def generate_cov_matrices(n, num_matrices=10000, seed=67):
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(num_matrices):
        A = rng.normal(size=(n, n))
        Sig = A @ A.T
        D   = np.sqrt(np.diag(Sig))
        corr = Sig / np.outer(D, D)  # correlation matrix

        # turn into covariance with random stds
        stds = rng.uniform(0.1, 0.3, size=n)
        cov  = corr * np.outer(stds, stds)
        matrices.append(cov)
    return matrices



#Randomize our matrices into "random" cov matrices using Wishart
#mild noise should be  noise factor of 10
def wishartize(mat, noise_factor):
    df = mat.shape[0] + noise_factor
    return sp.wishart.rvs(df=df, scale=mat/df)


# Construct our list of lists of matrices: (true mat, random mat)
def construct_data(noise_factor, n, num_matrices=10000, seed=67):
    simulated = []
    true_mats = generate_cov_matrices(n, num_matrices, seed)
    for mat in true_mats:
        noisy = wishartize(mat, noise_factor)
        simulated.append([mat, noisy])   # [true_cov, noisy_cov]
    return simulated


#takes in noisy wishart matrix, uses GBM to generate asset prices over a year. 
#mu is randomly chosen in a reasonable way
def generate_returns(mat):
    n = np.shape(mat)[0]
    mu = np.random.normal(0.1, 0.01, n) #this will put mus between 0.08 and 0.12 mostly
    times = np.linspace(1/252, 1, 252)
    price_mat = simulate_prices(mu, mat, times, seed=None)
    return price_mat

#generates true matrices, simulated Wishart matrices, does GBM, and gets corresponding sample covariance matrices
def construct_sample_cov(noise_factor=10, n=10, num_matrices=10000, seed=67):
    simulated = construct_data(noise_factor, n, num_matrices, seed)
    sample_mats = []
    for true_cov, noisy_cov in simulated:
        sample_cov = np.cov(generate_returns(noisy_cov), rowvar=False)
        sample_mats.append([true_cov, sample_cov])
    return sample_mats


