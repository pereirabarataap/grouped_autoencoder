import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin


class Encoder(nn.Module):
    """
    Linear encoder module with optional non-negativity constraints and structured regularization support.

    Parameters
    ----------
    in_features : int
        Number of input features (dimensionality of the input data).
    
    out_features : int
        Number of latent components to learn (dimensionality of the latent space).
    
    feature_classes : array-like of shape (n_features,), optional
        Array of class labels for each input feature. Determines how each feature is regularized:
            - np.nan: No regularization applied.
            - -1: Entropy regularization only.
            - >= 0: Both zero suppression and entropy regularization (class label indicates target component).
    
    non_negative : bool, default=True
        If True, constrains weights to be non-negative using an activation function (e.g. softplus or sigmoid).
    
    activation : str, default="sigmoid"
        Activation function used to enforce non-negativity:
            - "sigmoid": bounds weights between 0 and 1.
            - "softplus": unbounded positive weights.
            - Ignored if non_negative is False.
    
    w_init : str, default="rand"
        Weight initialization method:
            - "rand": Uniform random initialization [0, 1).
            - "randn": Standard normal initialization.
    """
    def __init__(self, in_features, out_features, feature_classes=None, non_negative=True, activation="sigmoid", w_init="rand"):
        super().__init__()

        # Initialize weight matrix
        if non_negative:
            if w_init == "rand":
                self.W_raw = nn.Parameter(torch.rand(in_features, out_features, dtype=torch.float32))
            elif w_init == "randn":
                self.W_raw = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float32))
            self.activation = F.sigmoid if activation == "sigmoid" else (
                F.softplus if activation == "softplus" else None
            )
        else:
            self.W_raw = nn.Parameter(torch.randn(in_features, out_features))
            self.activation = nn.Identity()

        self.non_negative = non_negative

        # Handle feature classes for regularization
        if feature_classes is not None:
            feature_classes = torch.tensor(feature_classes, dtype=torch.float32)

            # Mask logic:
            # - np.nan: no regularization
            # - -1: entropy only
            # - >= 0: zero + entropy regularization
            mask_nan = torch.isnan(feature_classes)
            mask_entropy = ~mask_nan
            mask_zero = (feature_classes >= 0) & ~mask_nan

            # Save class indices
            full_classes = torch.full((feature_classes.shape[0],), -2, dtype=torch.int64)
            full_classes[mask_entropy] = feature_classes[mask_entropy].to(torch.int64)
            self.register_buffer("feature_classes_full", full_classes)

            # Build zero regularization mask
            zero_mask = torch.zeros((in_features, out_features), dtype=torch.bool)
            for i, cls in enumerate(full_classes):
                if cls >= 0:
                    for j in range(out_features):
                        if j != cls:
                            zero_mask[i, j] = True
            self.register_buffer("mask_zero_entries", zero_mask)

            # Entropy regularization mask
            self.register_buffer("mask_entropy_entries", mask_entropy)
        else:
            self.register_buffer("feature_classes_full", torch.full((in_features,), -2, dtype=torch.int64))
            self.register_buffer("mask_zero_entries", torch.full((in_features, out_features), False, dtype=torch.bool))
            self.register_buffer("mask_entropy_entries", torch.full((in_features,), False, dtype=torch.bool))

    def forward(self, X):
        W = self.activation(self.W_raw)
        return X @ W


class Decoder(nn.Module):
    """
    Linear decoder that reconstructs input data from latent space using the transpose of encoder weights.

    Parameters
    ----------
    encoder : Encoder
        The encoder module whose weights are used for decoding.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, Z):
        W = self.encoder.activation(self.encoder.W_raw)
        return Z @ W.T


class GroupedAutoencoder(BaseEstimator, TransformerMixin):
    """
    A custom autoencoder with structured regularization for group-aware learning.

    Supports:
        - Sparse latent factors via L1/L2 (zero) regularization.
        - Entropy-based regularization for controlling feature spread across components.
        - Optional class-aware entropy grouping.
        - Non-negativity constraints for interpretability (e.g., parts-based learning).
    
    Parameters
    ----------
    theta : float, default=1.0
        Weighting factor for regularization vs. reconstruction error in total loss.
        If 1.0, only regularization is optimized; if 0.0, only reconstruction error.

    epsilon : float, default=0.5
        Balance between zero (sparsity) and entropy regularization terms.
        0.0 = only zero reg, 1.0 = only entropy reg, 0.5 = equal mix.

    l1_ratio : float, default=0.5
        Balance between L1 and L2 norms in zero regularization:
            - 1.0 = L1 only (sparsity).
            - 0.0 = L2 only (spread).
            - 0.5 = balanced.

    device : str, default='cpu'
        Torch device to use, e.g. 'cpu' or 'cuda'.

    w_init : str, default="rand"
        Weight initialization method for the encoder:
            - "rand": uniform [0, 1).
            - "randn": standard normal.

    verbose : bool or int, default=False
        If int > 0, prints training progress every `verbose` epochs.
        If False/0, disables logging.

    min_delta : float, default=1e-5
        Minimum loss improvement to be considered as progress (for early stopping and scheduler).

    n_components : int, default=2
        Dimensionality of latent space (number of components).

    random_state : int, default=42
        Random seed for reproducibility.

    max_iter : int, default=1e5
        Maximum number of training epochs.

    non_negative : bool, default=True
        If True, constrains encoder weights to be >= 0 using activation function.

    learning_rate : float, default=1e-1
        Initial learning rate for Adam optimizer.

    feature_classes : array-like of shape (n_features,), optional
        Metadata for each feature used to assign regularization types.
        See Encoder documentation for behavior of np.nan, -1, and class labels.

    scheduler_factor : float, default=0.5
        Factor by which the learning rate is reduced when no improvement is seen.

    activation : str, default="softplus"
        Non-negative activation function to use in the encoder:
            - "softplus": unbounded non-negative.
            - "sigmoid": [0, 1] bounded.

    entropy_scaling : str or None, default="exp"
        Scaling factor for entropy regularization term:
            - "log": 1 / log(n_components)
            - "exp": 1 / exp(n_components)
            - None: no scaling applied

    scheduler_patience : int, default=10
        Number of epochs with no improvement before reducing learning rate.

    entropy_on_classes : bool, default=False
        If True, entropy regularization is grouped by class label. Entropy is averaged within classes.

    early_stopping_patience : int, default=100
        Number of epochs with no improvement before stopping training early.
    """

    def __init__(
        self,
        theta=0.5,
        epsilon=0.5,
        l1_ratio=0.5,
        device='cpu',
        w_init="rand",
        verbose=False,
        min_delta=1e-5,
        n_components=2,
        random_state=42,
        max_iter=int(1e5),
        non_negative=True,
        learning_rate=1e-1,
        feature_classes=None,
        scheduler_factor=0.5,
        activation="softplus",
        entropy_scaling="exp",
        scheduler_patience=10,
        entropy_on_classes=False,
        early_stopping_patience=100,
    ):
        # Assign config
        self.device = device
        self.w_init = w_init
        
        self.verbose = int(verbose)
        self.activation = activation
        self.max_iter = int(max_iter)
        self.min_delta = float(min_delta)
        
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.non_negative = bool(non_negative)
        self.entropy_scaling = entropy_scaling
        self.feature_classes = feature_classes
        self.epsilon = np.clip(epsilon, 0.0, 1.0)
        self.learning_rate = float(learning_rate)
        self.l1_ratio = np.clip(l1_ratio, 0.0, 1.0)
        self.theta = np.clip(theta, 0.0, 1.0 - min_delta)
        self.scheduler_patience = int(scheduler_patience)
        self.entropy_on_classes = bool(entropy_on_classes)
        self.scheduler_factor = np.clip(scheduler_factor, 0.0, 1.0)
        self.early_stopping_patience = int(early_stopping_patience)
        
        
    def _build_model(self, input_dim):
        """
        Internal: Initializes encoder, decoder, and optimizer.
        """
        torch.manual_seed(self.random_state)
        self.encoder = Encoder(
            w_init=self.w_init,
            in_features=input_dim,
            activation=self.activation,
            out_features=self.n_components,
            non_negative=self.non_negative,
            feature_classes=self.feature_classes_,
        )
        self.decoder = Decoder(self.encoder)
        self.model = nn.Sequential(self.encoder, self.decoder).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def fit(self, X, X_val=None):
        """
        Trains the autoencoder on input data with optional validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input training data.

        X_val : array-like of shape (n_val_samples, n_features), optional
            Optional validation data for early stopping and loss tracking.

        Returns
        -------
        self : GroupedAutoencoder
            The fitted model.
        """
        # Training setup
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device) if X_val is not None else None
        n_features = X_tensor.shape[1]
    
        # Interpret special feature_classes shorthand
        if isinstance(self.feature_classes, (int, float)) and self.feature_classes == -1:
            self.feature_classes_ = np.full((n_features,), -1, dtype=np.float32)
        elif isinstance(self.feature_classes, (str, float)) and str(self.feature_classes).lower() == "nan":
            self.feature_classes_ = None
        elif self.feature_classes is not None:
            self.feature_classes_ = np.array(self.feature_classes, dtype=np.float32)
        else:
            self.feature_classes_ = None

        # Build model using parsed feature_classes
        self._build_model(X_tensor.shape[1])

        # Scaling factor for entropy
        if self.entropy_scaling is None:
            entropy_scaling = 1
        elif self.entropy_scaling == "log":
            entropy_scaling = 1 / np.log(self.n_components)
        elif self.entropy_scaling == "exp":
            entropy_scaling = 1 / np.exp(self.n_components)

        best_val = float('inf')
        epochs_no_improve = 0
        sched_epochs_no_improve = 0

        for epoch in range(self.max_iter):
            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass
            X_hat = self.model(X_tensor)
            rmse = torch.sqrt(self.loss_fn(X_hat, X_tensor))
            loss = rmse

            W = self.encoder.activation(self.encoder.W_raw)

            # --- Zero Regularization ---
            reg_struct = torch.tensor(0.0, device=W.device)
            if self.encoder.mask_zero_entries.any():
                mask = self.encoder.mask_zero_entries
                W_abs = W.abs()
                row_norm_l1 = W_abs / W_abs.sum(dim=1, keepdim=True).clamp_min(1e-12)
                row_norm_l2 = (W ** 2) / (W ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12)
                col_means_l1 = (row_norm_l1 * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)
                col_means_l2 = (row_norm_l2 * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)
                active_cols = mask.sum(dim=0) > 0
                if active_cols.any():
                    reg_struct_l1 = col_means_l1[active_cols].mean()
                    reg_struct_l2 = torch.sqrt(col_means_l2[active_cols].mean())
                    reg_struct = self.l1_ratio * reg_struct_l1 + (1 - self.l1_ratio) * reg_struct_l2

            # --- Entropy Regularization ---
            entropy_flag = 0
            reg_entropy = torch.tensor(0.0, device=W.device)
            probs_all = F.normalize(W.abs().clamp_min(1e-8), p=1, dim=1)
            entropies_all = -(probs_all * probs_all.log()).sum(dim=1)
            mask_entropy = self.encoder.mask_entropy_entries
            entropies_masked = entropies_all[mask_entropy]
            classes_masked = self.encoder.feature_classes_full[mask_entropy]

            valid_mask = classes_masked >= 0
            unknown_mask = classes_masked == -1
            grouped_mean = torch.tensor(0.0, device=W.device)
            if self.entropy_on_classes and valid_mask.any():
                valid_classes = classes_masked[valid_mask]
                valid_entropies = entropies_masked[valid_mask]
                unique_classes = torch.unique(valid_classes)
                group_sums = torch.zeros(unique_classes.max().item() + 1, device=W.device)
                group_counts = torch.zeros_like(group_sums)
                group_sums.index_add_(0, valid_classes, valid_entropies)
                group_counts.index_add_(0, valid_classes, torch.ones_like(valid_entropies))
                group_means = group_sums / group_counts.clamp_min(1)
                grouped_mean = group_means[group_counts > 0].mean()
                entropy_flag = 1
            unk_group_mean = torch.tensor(0.0, device=W.device)
            if unknown_mask.any():
                unk_group_mean = entropies_masked[unknown_mask].mean()
                entropy_flag = 1

            final_mean = grouped_mean + unk_group_mean
            if unknown_mask.any() and valid_mask.any() and self.entropy_on_classes:
                final_mean /= 2
            reg_entropy = final_mean * entropy_scaling

            # Combine regularizations
            reg_total = (1 - self.epsilon) * reg_struct + self.epsilon * reg_entropy if self.encoder.mask_zero_entries.any() and entropy_flag else reg_struct + reg_entropy
            loss = (1 - self.theta) * rmse + self.theta * reg_total

            loss.backward()
            self.optimizer.step()

            # Validation & Logging
            val_rmse, val_loss = rmse, loss
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_hat = self.model(X_val_tensor)
                    val_rmse = torch.sqrt(self.loss_fn(X_val_hat, X_val_tensor))
                    val_loss = (1 - self.theta) * val_rmse + self.theta * reg_total

            if self.verbose and (epoch % self.verbose == 0) and (epoch != 0):
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:5d} | NoImp: {epochs_no_improve:3d} | "
                    f"Train Loss: {loss.item():.5f} | Train Error: {rmse.item():.5f} | "
                    f"Val Loss: {val_loss.item():.5f} | Val Error: {val_rmse.item():.5f} | "
                    f"Zero Reg: {reg_struct.item():.5f} | Entropy Reg: {reg_entropy.item():.5f} | LR: {lr:.6f}",
                    end="\r", flush=True
                )

            # Early stopping and scheduler
            if val_loss < best_val - self.min_delta:
                best_val = val_loss
                epochs_no_improve = 0
                sched_epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                sched_epochs_no_improve += 1

            if sched_epochs_no_improve >= self.scheduler_patience:
                self.optimizer.param_groups[0]['lr'] *= self.scheduler_factor
                sched_epochs_no_improve = 0

            if epochs_no_improve >= self.early_stopping_patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break

        return self

    def transform(self, X):
        """
        Projects input data into the learned latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : ndarray of shape (n_samples, n_components)
            Latent representation of input.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            Z = self.encoder(X_tensor)
        return Z.cpu().numpy()

    def inverse_transform(self, Z):
        """
        Reconstructs data from latent representation.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components)
            Latent representations.

        Returns
        -------
        X_hat : ndarray of shape (n_samples, n_features)
            Reconstructed input data.
        """
        Z_tensor = torch.tensor(Z, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            X_hat = self.decoder(Z_tensor)
        return X_hat.cpu().numpy()

    def predict(self, X):
        """
        Full encode-decode process for reconstruction.

        Equivalent to: `inverse_transform(transform(X))`

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_hat : ndarray of shape (n_samples, n_features)
            Reconstructed input.
        """
        Z = self.transform(X)
        return self.inverse_transform(Z)

    def get_W(self, apply_scaling=False):
        """
        Retrieves the encoder's learned weight matrix W.

        Parameters
        ----------
        apply_scaling : bool, default=False
            If True, columns are normalized to unit L1 norm (sum to 1).

        Returns
        -------
        W : ndarray of shape (n_features, n_components)
            The weight matrix after activation and optional scaling.
        """
        W_tensor = self.encoder.activation(self.encoder.W_raw)
        if apply_scaling:
            W_tensor = F.normalize(W_tensor, p=1, dim=0)
        return W_tensor.detach().cpu().numpy()

    def to(self, device):
        """
        Moves the model and its parameters to the specified device.

        Parameters
        ----------
        device : str
            Target device (e.g. 'cuda' or 'cpu').

        Returns
        -------
        self : GroupedAutoencoder
            The model moved to the target device.
        """
        self.device = device
        if hasattr(self, "model"):
            self.model.to(device)
        return self
