import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util import construct_dataset
import os

# Enforce positive semi-definiteness on a matrix
def enforce_psd_torch (M, eps=1e-6):
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = torch.linalg.eigh(M)
    eigvals_clipped = torch.clamp(eigvals, eps, None)
    M_psd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T
    return M_psd

# Neural network model for covariance estimation
class CovarianceNet(nn.Module):
    
    def __init__(self, n_assets):
        super().__init__()
        self.n_assets = n_assets

        # Input: flattened returns matrix
        input_dim = n_assets * 252  # n_assets * lookback window
        
        # Output: upper triangular covariance elements
        output_dim = (n_assets * (n_assets + 1)) // 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, returns):
    
        batch_size = returns.shape[0]
        x = returns.reshape(batch_size, -1)
        upper_tri = self.network(x)
        
        # Reconstruct covariance matrices from upper triangular elements
        cov_matrices = []
        for i in range(batch_size):
            cov = self._build_symmetric_matrix(upper_tri[i])
            cov_matrices.append(enforce_psd_torch(cov))
        
        return torch.stack(cov_matrices)
    
    # Build symmetric matrix from upper triangular elements
    def _build_symmetric_matrix(self, upper_tri_elements):
        cov = torch.zeros(self.n_assets, self.n_assets, device=upper_tri_elements.device)
        idx = 0
        for i in range(self.n_assets):
            for j in range(i, self.n_assets):
                cov[i, j] = upper_tri_elements[idx]
                cov[j, i] = upper_tri_elements[idx]
                idx += 1
        return cov

# PyTorch dataset for covariance estimation
class CovarianceDataset(Dataset):
    def __init__(self, n_samples, n_assets):
        print(f"Generating {n_samples} training samples...")
        output = construct_dataset(noise_factor=10, n=n_assets, num_matrices=n_samples, seed=67)
        self.returns_list = [r.flatten() for _, _, r in output]
        self.cov_list = [c for c, _, _ in output]
    
    def __len__(self):
        return len(self.returns_list)
    
    def __getitem__(self, idx):
        returns = torch.FloatTensor(self.returns_list[idx])  # (lookback, n_assets)
        cov = torch.FloatTensor(self.cov_list[idx])  # (n_assets, n_assets)
        return returns, cov

# Train the covariance estimation neural network on simulated covariance data
def train_covariance_model(
    n_assets = 10,
    n_train_samples = 2000,
    n_val_samples = 500,
    batch_size = 64,
    epochs = 25,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    
    print(f"Training on device: {device}")
    
    # Create datasets
    train_dataset = CovarianceDataset(n_train_samples, n_assets)
    val_dataset = CovarianceDataset(n_val_samples, n_assets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and optimizer
    model = CovarianceNet(n_assets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function is frobenius norm
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("\nTraining:")
    print("="*60)
    
    for epoch in range(epochs):
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_returns, batch_cov in train_loader:
            batch_returns = batch_returns.to(device)
            batch_cov = batch_cov.to(device)
            
            # Forward pass
            pred_cov = model(batch_returns)
            loss = criterion(pred_cov, batch_cov)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_returns, batch_cov in val_loader:
                batch_returns = batch_returns.to(device)
                batch_cov = batch_cov.to(device)
                
                pred_cov = model(batch_returns)
                loss = criterion(pred_cov, batch_cov)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    print("="*60)
    return model

# Predict covariance matrix from returns using trained model
def predict_covariance(model, returns, device='cpu'):
    model.eval()
    with torch.no_grad():
        returns_tensor = torch.FloatTensor(returns).unsqueeze(0).to(device)
        pred_cov = model(returns_tensor)
        return (pred_cov).squeeze(0).cpu().numpy()

if __name__ == "__main__":
    
    n_assets = 10 # number of assets
    
    # Train the model
    model = train_covariance_model(
        n_assets=n_assets,
        n_train_samples=1600,
        n_val_samples=400,
        batch_size=64,
        epochs=20,
        learning_rate=0.0001
    )
    
    # Test predictions
    print("\nTesting:")
    errors = []
    data = construct_dataset(10, n_assets, 500)
    for i in range(len(data)):
        test_cov = data[i][0]
        test_returns = data[i][2]
        pred_cov = predict_covariance(model, test_returns)
        errors.append(np.linalg.norm(test_cov - pred_cov))
    print("Mean Frobenius Error:", np.mean(errors))
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/neural_network.pt')
    print("Model saved to models/neural_network.pt")