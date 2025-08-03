import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Encoder(nn.Module):
    """
    Encoder module for the Grouped Autoencoder.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of latent components.
        feature_classes (list or None): Optional list assigning class labels to input features.
        non_negative (bool): If True, applies sigmoid to ensure non-negative weights.
    """
    def __init__(self, in_features, out_features, feature_classes=None, non_negative=True):
        super().__init__()
        self.W_raw = nn.Parameter(torch.randn(in_features, out_features))
        self.non_negative = non_negative

        if feature_classes is not None:
            feature_classes = torch.tensor(feature_classes, dtype=torch.int64)
            zero_mask = torch.ones((in_features, out_features), dtype=torch.bool)
            zero_mask[feature_classes == 0, 0] = False
            zero_mask[feature_classes == 1, 1] = False
            self.register_buffer("mask_zero_entries", zero_mask)
            self.register_buffer("mask_0", feature_classes == 0)
            self.register_buffer("mask_1", feature_classes == 1)
        else:
            self.register_buffer("mask_zero_entries", None)
            self.register_buffer("mask_0", None)
            self.register_buffer("mask_1", None)

    def get_W(self, theta=1.0):
        W_source = torch.sigmoid(self.W_raw) if self.non_negative else self.W_raw

        if self.mask_zero_entries is None or theta < 1.0:
            return W_source

        W = torch.zeros_like(self.W_raw)
        W[self.mask_0, 0] = W_source[self.mask_0, 0]
        W[self.mask_1, 1] = W_source[self.mask_1, 1]
        return W

    def forward(self, X, theta=1.0):
        W = self.get_W(theta)
        return X @ W


class Decoder(nn.Module):
    """
    Decoder module for the Grouped Autoencoder.

    Args:
        encoder (Encoder): Encoder instance whose weights are shared.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, Z, theta=1.0):
        W = self.encoder.get_W(theta)
        return Z @ W.T


class GroupedAutoencoder(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible grouped autoencoder with optional feature grouping and regularization.

    Args:
        n_components (int): Dimensionality of latent space.
        max_iter (int): Maximum training iterations.
        learning_rate (float): Initial learning rate.
        early_stopping_patience (int): Epochs to wait before stopping on plateau.
        min_delta (float): Minimum improvement to reset patience.
        device (str): 'cpu' or 'cuda'.
        feature_classes (list or None): Optional feature class assignments.
        non_negative (bool): If True, constrain encoder weights to be non-negative.
        verbose (bool): If True, prints progress.
        random_state (int): Random seed.
        scheduler_patience (int): Learning rate scheduler patience.
        theta (float): Weighting for regularization vs reconstruction.
    """
    def __init__(
        self,
        n_components=2,
        max_iter=10000,
        learning_rate=1e-1,
        early_stopping_patience=200,
        min_delta=1e-5,
        device='cpu',
        feature_classes=None,
        non_negative=False,
        verbose=False,
        random_state=42,
        scheduler_patience=100,
        theta=1.0
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.device = device
        self.feature_classes = feature_classes
        self.non_negative = non_negative
        self.verbose = verbose
        self.random_state = random_state
        self.scheduler_patience = scheduler_patience
        self.theta = theta

    def _build_model(self, input_dim):
        torch.manual_seed(self.random_state)
        self.encoder = Encoder(
            in_features=input_dim,
            out_features=self.n_components,
            feature_classes=self.feature_classes,
            non_negative=self.non_negative
        )
        self.decoder = Decoder(self.encoder)
        self.model = nn.Sequential(self.encoder, self.decoder).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.scheduler_patience,
            min_lr=1e-8
        )

    def fit(self, X, X_val=None):
        """
        Train the autoencoder.

        Args:
            X (ndarray): Training data.
            X_val (ndarray or None): Optional validation data.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device) if X_val is not None else None

        self._build_model(X_tensor.shape[1])

        best_val = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.max_iter):
            self.model.train()
            self.optimizer.zero_grad()

            Z = self.encoder(X_tensor, theta=self.theta)
            X_hat = self.decoder(Z, theta=self.theta)
            rmse = torch.sqrt(self.loss_fn(X_hat, X_tensor))
            loss = rmse

            if self.feature_classes is not None and 0 < self.theta < 1:
                W = torch.sigmoid(self.encoder.W_raw) if self.non_negative else self.encoder.W_raw
                reg_mask = self.encoder.mask_zero_entries
                reg = torch.sum(W[reg_mask] ** 2)
                loss = (1 - self.theta) * rmse + self.theta * reg

            elif self.feature_classes is None:
                W = torch.sigmoid(self.encoder.W_raw) if self.non_negative else self.encoder.W_raw
                probs = F.normalize(W.abs(), p=2, dim=1)
                reg = torch.mean(torch.distributions.Categorical(probs=probs).entropy()) / (np.exp(self.n_components))
                loss = (1 - self.theta) * rmse + self.theta * reg

            loss.backward()
            self.optimizer.step()

            val_rmse = rmse
            val_loss = loss
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    Z_val = self.encoder(X_val_tensor, theta=self.theta)
                    X_val_hat = self.decoder(Z_val, theta=self.theta)
                    val_rmse = self.loss_fn(X_val_hat, X_val_tensor)
                    val_loss = torch.sqrt(val_rmse)

            self.scheduler.step(val_loss)

            if self.verbose and epoch % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:5d} | Train Error: {rmse.item():.5f} | Val Error: {val_rmse.item():.5f} | LR: {lr:.6f}",
                    end="\r", flush=True
                )

            if val_loss < best_val - self.min_delta:
                best_val = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.early_stopping_patience:
                if self.verbose:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(
                        f"\nEarly stopping at epoch {epoch:5d} | Train Error: {rmse.item():.5f} | Val Error: {val_rmse.item():.5f} | LR: {lr:.6f}"
                    )
                break

        return self

    def transform(self, X):
        """Encode input X to latent space."""
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            Z = self.encoder(X_tensor, theta=self.theta)
        return Z.cpu().numpy()

    def inverse_transform(self, Z):
        """Decode latent vector Z back to input space."""
        Z_tensor = torch.tensor(Z, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            X_hat = self.decoder(Z_tensor, theta=self.theta)
        return X_hat.cpu().numpy()

    def predict(self, X):
        """Encode and decode input X."""
        Z = self.transform(X)
        return self.inverse_transform(Z)

    def get_W(self):
        """Return learned weight matrix W."""
        W_tensor = self.encoder.get_W(theta=self.theta)
        if apply_scaling:
            # L1 norm along rows (each column must sum to 1)
            W_tensor = F.normalize(W_tensor, p=1, dim=0)
        return W_tensor.detach().cpu().numpy()
