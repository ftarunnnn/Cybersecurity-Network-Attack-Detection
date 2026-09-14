import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ---------------------------------------------------------
# 1. 1D CNN Architecture (Feature Pattern Detection)
# ---------------------------------------------------------
class NetworkCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for spatial packet feature pattern extraction.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super(NetworkCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc1 = nn.Linear(64 * 4, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch, input_dim] -> unsqueeze to [batch, 1, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.fc2(out)
        return out


# ---------------------------------------------------------
# 2. LSTM Architecture (Sequential / Time-based Network Behavior)
# ---------------------------------------------------------
class NetworkLSTM(nn.Module):
    """
    Long Short-Term Memory Network for sequence and temporal packet behavior analysis.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2):
        super(NetworkLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len=1, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use final time-step hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out


# ---------------------------------------------------------
# 3. Autoencoder Architecture (Unsupervised Anomaly Detection)
# ---------------------------------------------------------
class NetworkAutoencoder(nn.Module):
    """
    Deep Autoencoder trained on Normal traffic to measure reconstruction error
    for zero-day anomaly detection.
    """
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super(NetworkAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_reconstruction_error(self, x):
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse.numpy()


# ---------------------------------------------------------
# 4. DL Training & Evaluation Manager
# ---------------------------------------------------------
class DLTrainerManager:
    def __init__(self, input_dim: int, num_classes: int, device: str = 'cpu'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = torch.device(device)
        
        self.cnn_model = NetworkCNN1D(input_dim, num_classes).to(self.device)
        self.lstm_model = NetworkLSTM(input_dim, hidden_dim=32, num_classes=num_classes).to(self.device)
        self.autoencoder = NetworkAutoencoder(input_dim, latent_dim=4).to(self.device)
        self.ae_threshold = 0.5

    def train_classifier(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 15, lr: float = 0.001):
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bx.size(0)

    def train_autoencoder(self, X_normal: np.ndarray, epochs: int = 20, lr: float = 0.001):
        X_t = torch.tensor(X_normal, dtype=torch.float32)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)

        self.autoencoder.train()
        for epoch in range(epochs):
            for (bx,) in loader:
                bx = bx.to(self.device)
                optimizer.zero_grad()
                recon = self.autoencoder(bx)
                loss = criterion(recon, bx)
                loss.backward()
                optimizer.step()

        # Compute 95th percentile reconstruction error on normal traffic as anomaly threshold
        self.autoencoder.eval()
        with torch.no_grad():
            recon_normal = self.autoencoder(X_t.to(self.device))
            errors = torch.mean((X_t.to(self.device) - recon_normal) ** 2, dim=1).cpu().numpy()
            self.ae_threshold = float(np.percentile(errors, 95))
            print(f"[Autoencoder] Set 95th percentile anomaly threshold: {self.ae_threshold:.4f}")

    def predict_with_confidence(self, model: nn.Module, X: np.ndarray):
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
        return preds, confidences

    def save_all(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.cnn_model.state_dict(), os.path.join(save_dir, 'cnn_model.pt'))
        torch.save(self.lstm_model.state_dict(), os.path.join(save_dir, 'lstm_model.pt'))
        torch.save({
            'state_dict': self.autoencoder.state_dict(),
            'threshold': self.ae_threshold
        }, os.path.join(save_dir, 'autoencoder_model.pt'))
        print(f"[OK] PyTorch Deep Learning models saved to: {save_dir}")

    def load_all(self, save_dir: str):
        cnn_path = os.path.join(save_dir, 'cnn_model.pt')
        lstm_path = os.path.join(save_dir, 'lstm_model.pt')
        ae_path = os.path.join(save_dir, 'autoencoder_model.pt')

        if os.path.exists(cnn_path):
            self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
        if os.path.exists(lstm_path):
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
        if os.path.exists(ae_path):
            checkpoint = torch.load(ae_path, map_location=self.device)
            self.autoencoder.load_state_dict(checkpoint['state_dict'])
            self.ae_threshold = checkpoint.get('threshold', 0.5)
