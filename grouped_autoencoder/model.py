import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin

class Encoder(nn.Module):
    def __init__(self, in_features, out_features, feature_classes=None, non_negative=True):
        super().__init__()

        if non_negative:
            self.W_raw = nn.Parameter(torch.rand(in_features, out_features))
            self.activation = F.softplus
        else:
            self.W_raw = nn.Parameter(torch.randn(in_features, out_features))
            self.activation = nn.Identity()

        self.non_negative = non_negative
        
        if feature_classes is not None:
            feature_classes = torch.tensor(feature_classes, dtype=torch.float32)
            nan_mask = torch.isnan(feature_classes)
        
            if nan_mask.all():
                self.register_buffer("feature_classes_full", torch.full((in_features,), -1, dtype=torch.int64))
                self.register_buffer("mask_zero_entries", None)
                self.register_buffer("nan_mask", nan_mask)
            else:
                valid_classes = feature_classes[~nan_mask].to(torch.int64)
                unique_classes = valid_classes.unique()
                num_nan = nan_mask.sum().item()
                num_classes = len(unique_classes)

                if num_nan == 0:
                    expected_classes = set(range(out_features))
                    present_classes = set(valid_classes.tolist())
                    missing_classes = expected_classes - present_classes
                    assert not missing_classes, (
                        f"feature_classes must contain all class indices 0 through {out_features - 1}. "
                        f"Missing: {missing_classes}"
                    )
                
                assert num_classes + num_nan >= out_features, (
                    f"Expected at least {out_features} total classes (non-NaN + NaN), got {num_classes} + {num_nan}"
                )
        
                full_classes = torch.full((feature_classes.shape[0],), -1, dtype=torch.int64)
                full_classes[~nan_mask] = valid_classes
                self.register_buffer("feature_classes_full", full_classes)
                self.register_buffer("nan_mask", nan_mask)
        
                zero_mask = torch.zeros((in_features, out_features), dtype=torch.bool)
                for i, cls in enumerate(full_classes):
                    if cls >= 0:  # only for known (non-NaN) classes
                        for j in range(out_features):
                            if j != cls:
                                zero_mask[i, j] = True
                self.register_buffer("mask_zero_entries", zero_mask)
                
        else:
            self.register_buffer("feature_classes_full", None)
            self.register_buffer("mask_zero_entries", None)
            self.register_buffer("nan_mask", None)

    def forward(self, X, theta=1.0):
        W = self.activation(self.W_raw)
        return X @ W

class Decoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, Z, theta=1.0):
        W = self.encoder.activation(self.encoder.W_raw)
        return Z @ W.T

class GroupedAutoencoder(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        n_components=2,
        max_iter=int(1e5),
        learning_rate=1e-1,
        early_stopping_patience=200,
        min_delta=1e-5,
        device='cpu',
        feature_classes=None,
        non_negative=True,
        verbose=False,
        random_state=42,
        scheduler_patience=100,
        l1_ratio=0.5,
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
        self.verbose = int(verbose)
        self.random_state = random_state
        self.scheduler_patience = scheduler_patience
        self.l1_ratio = np.clip(l1_ratio, 0,1)
        self.theta = np.clip(theta, 0, 1)
        

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
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device) if X_val is not None else None
        
        self._build_model(X_tensor.shape[1])

        # Prevents full disregard of reconstruction loss when theta == 1
        loss_theta = (1 - 1e-4) if self.theta == 1 else self.theta
        entropy_scaling = 1 / np.exp(self.n_components)
        best_val = float('inf')
        epochs_no_improve = 0
        for epoch in range(self.max_iter):
            self.model.train()
            self.optimizer.zero_grad()

            Z = self.encoder(X_tensor, theta=self.theta)
            X_hat = self.decoder(Z, theta=self.theta)
            rmse = torch.sqrt(self.loss_fn(X_hat, X_tensor))
            loss = rmse

            W = self.encoder.activation(self.encoder.W_raw)
            
            # Regularization
            reg_struct = torch.Tensor([0.0])
            reg_entropy = torch.Tensor([0.0])
        
            # Zeros Regularization
            if self.encoder.mask_zero_entries is not None:
                # columns vector W-balanced L2 regularisation
                mask = self.encoder.mask_zero_entries
                col_sums_l1 = (W.abs() * mask).sum(dim=0)
                col_sums_l2 = (W**2 * mask).sum(dim=0)
                col_counts = mask.sum(dim=0)
                col_means_l1 = col_sums_l1 / col_counts.clamp(min=1)
                col_means_l2 = col_sums_l2 / col_counts.clamp(min=1)
                active_cols = col_counts > 0
                if active_cols.any():
                    reg_struct_l1 = col_means_l1[active_cols].mean()
                    reg_struct_l2 = torch.sqrt(col_means_l2[active_cols].mean())
                    reg_struct = self.l1_ratio * reg_struct_l1 + (1 - self.l1_ratio) * reg_struct_l2

            # Entropy Regularization
            if self.encoder.nan_mask is None or self.encoder.nan_mask.any():
                if self.encoder.nan_mask is not None:
                    W_nan = W[self.encoder.nan_mask]
                else:
                    W_nan = W
                if W_nan.numel() > 0:
                    probs = F.normalize(W_nan.abs().clamp_min(1e-8), p=1, dim=1)
                    reg_entropy = torch.mean(torch.distributions.Categorical(probs=probs).entropy()) * entropy_scaling
            
            reg_total = reg_struct + reg_entropy
            loss = (1 - loss_theta) * rmse + loss_theta * reg_total

            # Validation
            val_rmse = rmse
            val_loss = loss
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    Z_val = self.encoder(X_val_tensor, theta=self.theta)
                    X_val_hat = self.decoder(Z_val, theta=self.theta)
                    val_rmse = torch.sqrt(self.loss_fn(X_val_hat, X_val_tensor))
                    val_loss = (1 - loss_theta) * val_rmse + loss_theta * reg_total

            if self.verbose and (epoch % self.verbose == 0) and (epoch!=0):
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:5d} | "
                    f"Train Loss: {loss.item():.5f} | Train Error: {rmse.item():.5f} | "
                    f"Val Loss: {val_loss.item():.5f} | Val Error: {val_rmse.item():.5f} | "
                    f"Zero Reg: {reg_struct.item():.5f} | Entropy Reg: {reg_entropy.item():.5f} | LR: {lr:.6f}",
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
                        f"EStop {epoch:5d} | "
                        f"Train Loss: {loss.item():.5f} | Train Error: {rmse.item():.5f} | "
                        f"Val Loss: {val_loss.item():.5f} | Val Error: {val_rmse.item():.5f} | "
                        f"Zero Reg: {reg_struct.item():.5f} | Entropy Reg: {reg_entropy.item():.5f} | LR: {lr:.6f}",
                        f"\n", end="\r", flush=True
                    )
                break

            if epoch==(self.max_iter-1) and self.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"NoCnv {epoch:5d} | "
                    f"Train Loss: {loss.item():.5f} | Train Error: {rmse.item():.5f} | "
                    f"Val Loss: {val_loss.item():.5f} | Val Error: {val_rmse.item():.5f} | "
                    f"Zero Reg: {reg_struct.item():.5f} | Entropy Reg: {reg_entropy.item():.5f} | LR: {lr:.6f}",
                    f"\n", end="\r", flush=True
                )
        
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(val_loss)
        
        return self
            
    def transform(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            Z = self.encoder(X_tensor, theta=self.theta)
        return Z.cpu().numpy()

    def inverse_transform(self, Z):
        Z_tensor = torch.tensor(Z, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            X_hat = self.decoder(Z_tensor, theta=self.theta)
        return X_hat.cpu().numpy()

    def predict(self, X):
        Z = self.transform(X)
        return self.inverse_transform(Z)

    def get_W(self, apply_scaling=False):
        W_tensor = self.encoder.activation(self.encoder.W_raw)
        if apply_scaling:
            W_tensor = F.normalize(W_tensor, p=1, dim=0)
        return W_tensor.detach().cpu().numpy()

    def to(self, device):
        self.device = device
        if hasattr(self, "model"):
            self.model.to(device)
        return self

