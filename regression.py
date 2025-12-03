import numpy as np
import sklearn as sk
from generate_data import construct_sample_cov

def flatten_cov(mat):
    idx = np.triu_indices_from(mat)
    return mat[idx]

# Make our data
data = construct_sample_cov(num_matrices=10000)

# Train/test split
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data  = data[split_idx:]

# Build regression matrices
X = []
y = []

for true_cov, sample_cov in train_data:
    X.append(flatten_cov(sample_cov))
    y.append(flatten_cov(true_cov))

X = np.array(X)
y = np.array(y)

# Train model
model = sk.linear_model.LinearRegression()
model.fit(X, y)

print('model trained')
print("X shape:", X.shape)
print("y shape:", y.shape)


X_test = []
y_test = []

# Test model

for true_cov, sample_cov in test_data:
    X_test.append(flatten_cov(sample_cov))
    y_test.append(flatten_cov(true_cov))

X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred = model.predict(X_test)

# check MSE

mse = sk.metrics.mean_squared_error(y_test, y_pred)
print("MSE on covariance entries:", mse)

def unflatten(v, n):
    M = np.zeros((n, n))
    idx = np.triu_indices(n)
    M[idx] = v
    M[(idx[1], idx[0])] = v
    return M

n = true_cov.shape[0]   # number of assets

fro_errors = []
for pred_vec, true_vec in zip(y_pred, y_test):
    pred = unflatten(pred_vec, n)
    true = unflatten(true_vec, n)
    fro_errors.append(np.linalg.norm(pred - true, 'fro'))

fro_mean = np.mean(fro_errors)
print("Mean Frobenius norm error:", fro_mean)

# check if outperforms sample cov

sample_errors = []
learned_errors = []

for (true_cov, sample_cov), pred_vec in zip(test_data, y_pred):
    pred_cov = unflatten(pred_vec, n)

    sample_errors.append(np.linalg.norm(sample_cov - true_cov, 'fro'))
    learned_errors.append(np.linalg.norm(pred_cov - true_cov, 'fro'))

print("Sample cov Frobenius error:", np.mean(sample_errors))
print("Learned cov Frobenius error:", np.mean(learned_errors))

