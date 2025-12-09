import numpy as np
from util import flatten_cov, unflatten_cov, enforce_psd, frobenius_error
from generate_data import construct_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Builds dataset for training random forest model for covariance estimation
def build_rf_dataset(num_matrices=1000, n=10, T=252, seed=67):
    dataset = construct_dataset(noise_factor=10, n=n, num_matrices=num_matrices, seed=seed)
    X = [r.flatten() for _, _, r in dataset]
    y = [flatten_cov(c) for c, _, _ in dataset]
    return np.array(X), np.array(y)

# Build dataset
X, y = build_rf_dataset(num_matrices=2000, n=10, T=252)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Build model
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    n_jobs=-1,
    random_state=0,
    verbose=False
)

# Fit model
rf.fit(X_train, y_train)
print("Trained Random Forest on", X_train.shape[0], "samples.")

# Validate model on test dataset
y_pred = rf.predict(X_test)
error = np.mean([frobenius_error(p, t) for p, t in zip(y_pred, y_test)])
print("Mean Frobenius Error (RF):", error)

# Create covariance estimator function using trained RF and PSD enforcement
def rf_cov_estimator(model, returns):
    pred_vec = model.predict(returns.reshape(1, -1))[0]
    M = unflatten_cov(pred_vec, n=10)
    M_psd = enforce_psd(M)
    return M_psd

# Evaluate model using rf_cov_estimator
y_pred_psd = [flatten_cov(rf_cov_estimator(rf, returns)) for returns in X_test]
error = np.mean([frobenius_error(pred, test) for pred, test in zip(y_pred_psd, y_test)])
print("Mean Frobenius Error (RF w/ PSD):", error)

# Save model
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("Saved Random Forest Model to models/random_forest.pkl")