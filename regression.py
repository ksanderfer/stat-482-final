import numpy as np
from util import flatten_cov, unflatten_cov, enforce_psd, frobenius_error, build_training_dataset, sample_covariance, ledoit_wolf_shrinkage
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf
import pickle
import os

# Create covariance estimator function using trained sklearn model and PSD enforcement
def cov_estimator(model, returns):
    pred_vec = model.predict(returns.reshape(1, -1))[0]
    M = unflatten_cov(pred_vec, n=10)
    M_psd = enforce_psd(M)
    return M_psd

def main ():
    # Build dataset
    X, y = build_training_dataset(num_matrices=2000, n=10, T=252)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Build models
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        n_jobs=-1,
        random_state=0,
        verbose=False
    )
    lr = LinearRegression()

    # Fit models
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    print("Trained models on", X_train.shape[0], "samples.")

    # Validate models on test dataset
    y_pred = rf.predict(X_test)
    error = np.mean([frobenius_error(p, t) for p, t in zip(y_pred, y_test)])
    print("Mean Frobenius Error (Random Forest):", error)

    y_pred = lr.predict(X_test)
    error = np.mean([frobenius_error(p, t) for p, t in zip(y_pred, y_test)])
    print("Mean Frobenius Error (Linear Regression):", error)

    # Evaluate models using cov_estimator
    y_pred_psd = [flatten_cov(cov_estimator(rf, returns)) for returns in X_test]
    error = np.mean([frobenius_error(pred, test) for pred, test in zip(y_pred_psd, y_test)])
    print("Mean Frobenius Error (RF w/ PSD):", error)

    y_pred_psd = [flatten_cov(cov_estimator(lr, returns)) for returns in X_test]
    error = np.mean([frobenius_error(pred, test) for pred, test in zip(y_pred_psd, y_test)])
    print("Mean Frobenius Error (Linear Regression w/ PSD):", error)

    # Compare against sample covariance and Ledoit-Wolf shrinkage
    y_pred = [flatten_cov(sample_covariance(returns.reshape(-1, 10))) for returns in X_test]
    error = np.mean([frobenius_error(pred, test) for pred, test in zip(y_pred, y_test)])
    print("Mean Frobenius Error (Sample Covariance):", error)

    y_pred = [flatten_cov(ledoit_wolf_shrinkage(returns.reshape(-1, 10))) for returns in X_test]
    error = np.mean([frobenius_error(pred, test) for pred, test in zip(y_pred, y_test)])
    print("Mean Frobenius Error (Ledoit-Wolf Shrinkage):", error)

    # Save models
    if not os.path.exists('models'):
        os.makedirs('models')

    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print("Saved Random Forest Model to models/random_forest.pkl")

    with open('models/linear_regression.pkl', 'wb') as f:
        pickle.dump(lr, f)
    print("Saved Linear Regression Model to models/linear_regression.pkl")

if __name__ == "__main__":
    main()