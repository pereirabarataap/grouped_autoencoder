import os
import copy
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
    
    activation : str, default="softplus"
        Activation function used to enforce non-negativity:
            - "sigmoid": bounds weights between 0 and 1.
            - "softplus": unbounded positive weights.
            - Ignored if non_negative is False.
    
    w_init : str, default="randn"
        Weight initialization method:
            - "rand": Uniform random initialization [0, 1).
            - "randn": Standard normal initialization.
    """
    def __init__(self, in_features, out_features, feature_classes=None, non_negative=True, activation="softplus", w_init="randn"):
        super().__init__()
 
        # --- Encoder: activation & w_init guard ---
        if non_negative:
            # Pick activation function and optimal W_raw center (m) for max derivative
            if activation == "sigmoid":
                self.activation = torch.sigmoid
                m = 0.0  # sigmoid'(x) peaks at x = 0
            elif activation == "softplus":
                self.activation = F.softplus
                m = np.log(np.e - 1)  # ~0.5413, center for softplus'(x) ≈ max
            else:
                raise ValueError('activation must be "sigmoid" or "softplus"')
        
            # Initialize weight matrix in high-derivative region
            if (w_init is None) or (w_init == "rand"):
                # Small random spread around m for symmetry breaking
                self.W_raw = nn.Parameter(
                    torch.empty(in_features, out_features, dtype=torch.float32).uniform_(m - 0.1, m + 0.1)
                )
            elif w_init == "randn":
                self.W_raw = nn.Parameter(
                    torch.randn(in_features, out_features, dtype=torch.float32) * 0.05 + m
                )
            else:
                self.W_raw = nn.Parameter(torch.tensor(w_init, dtype=torch.float32))
        else:
            self.W_raw = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float32))
            self.activation = nn.Identity()

        self.non_negative = non_negative

        # Handle feature classes for regularization
        if feature_classes is not None:
            fc = torch.tensor(feature_classes, dtype=torch.float32)
        
            # masks
            mask_nan = torch.isnan(fc)
            mask_entropy = ~mask_nan                      # entropy on all non-NaN (>=0 and -1)
            mask_zero = (~mask_nan) & (fc >= 0)           # zero-reg only for known classes
        
            # store class indices: -2 = NaN, -1 = entropy-only, >=0 = class id
            full_classes = torch.full((fc.shape[0],), -2, dtype=torch.int64)
            full_classes[~mask_nan] = fc[~mask_nan].to(torch.int64)
            self.register_buffer("feature_classes_full", full_classes)
        
            # --- vectorized zero off-group mask ---
            d, k = in_features, out_features
            zero_mask = torch.zeros((d, k), dtype=torch.bool)
            idx_rows = torch.nonzero(full_classes >= 0, as_tuple=False).squeeze(1)
            if idx_rows.numel() > 0:
                zero_mask[idx_rows, :] = True
                idx_cols = full_classes[idx_rows].clamp(min=0, max=k-1)  # optional safety
                zero_mask[idx_rows, idx_cols] = False
            self.register_buffer("mask_zero_entries", zero_mask)
        
            # entropy mask per feature (1D)
            self.register_buffer("mask_entropy_entries", mask_entropy)
        else:
            self.register_buffer("feature_classes_full", torch.full((in_features,), -2, dtype=torch.int64))
            self.register_buffer("mask_zero_entries", torch.zeros((in_features, out_features), dtype=torch.bool))
            self.register_buffer("mask_entropy_entries", torch.zeros((in_features,), dtype=torch.bool))

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

class CLEAR(BaseEstimator, TransformerMixin):
    """
    A custom autoencoder with structured regularization for group-aware learning.
        
    Supports:
        - Sparse latent factors via L1/L2 (zero) regularization.
        - Entropy-based regularization for controlling feature spread across components.
        - Optional class-aware entropy grouping.
        - Non-negativity constraints for interpretability (e.g., parts-based learning).
        - Baseline normalization of loss terms for balanced optimization.
        - Gradual warm-up of the regularization–reconstruction trade-off, with independent schedules
          for zero and entropy regularization.
        
    Parameters
    ----------
    theta : float, default=0.5
        Weighting factor for regularization vs. reconstruction error in total loss.
        Internally clipped to be less than 1.0.
        If 1.0, only regularization is optimized; if 0.0, only reconstruction error.
    
    epsilon : float, default=0.5
        Balance between zero (sparsity) and entropy regularization terms.
        0.0 = only zero reg, 1.0 = only entropy reg, 0.5 = equal mix.
        If only one regularization type is active, it receives the full `theta` budget.
    
    l1_ratio : float, default=0.5
        Balance between L1 and L2 norms in zero regularization:
            - 1.0 = L1 only (sparsity).
            - 0.0 = L2 only (spread).
            - 0.5 = balanced.
    
    device : str, default='cpu'
        Torch device to use, e.g. 'cpu' or 'cuda'.
    
    w_init : str, default=None
        Weight initialization method for the encoder:
            - None: defaults to "rand" if non_negative==True else "randn"
            - "rand": uniform [0, 1).
            - "randn": standard normal.
            - array-like: custom initialization.
    
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
    
    theta_warmup : int, default=None
        Number of epochs to gradually increase `theta` from 0 to its target value.
        If set to None, will be inferred from X upon fitting: sqrt(X.shape[0] * X.shape[1])
    
    zero_warmup : {"linear", "bezier_convex", "bezier_concave"}, default="linear"
        Warm-up schedule for zero regularization:
            - "linear": increases proportionally to training progress.
            - "bezier_convex": slow start, accelerates towards the end.
            - "bezier_concave": fast start, slows down towards the end.
    
    entropy_warmup : {"linear", "bezier_convex", "bezier_concave"}, default="bezier_convex"
        Warm-up schedule for entropy regularization (same options as `zero_warmup`).
    
    non_negative : bool, default=False
        If True, constrains encoder weights to be >= 0 using an activation function.
    
    learning_rate : float, default=1e-1
        Initial learning rate for Adam optimizer.
    
    feature_classes : array-like of shape (n_features,), optional
        Metadata for each feature used to assign regularization types.
        See `Encoder` documentation for behavior of `np.nan`, `-1`, and non-negative class labels.
    
    scheduler_factor : float, default=0.5
        Factor by which the learning rate is reduced when no improvement is seen.
    
    activation : str, default="softplus"
        Non-negative activation function to use in the encoder when non_negative==True:
            - "softplus": unbounded non-negative.
            - "sigmoid": [0, 1] bounded.
    
    entropy_scaling : {'log', 'exp'} or float or None, default="log"
        Scaling factor for entropy regularization term:
            - "log": 1 / log(n_components)  (safe for n_components=1)
            - "exp": 1 / exp(n_components)
            - float: custom scaling factor
            - None: no scaling applied
    
    scheduler_patience : int, optional
        Number of consecutive epochs with no improvement in validation loss
        before reducing the learning rate.
        If ``None``, this is set automatically based on the warm-up length:
        ``max(200, theta_warmup // 50)``.
        A shorter patience than warm-up ensures the learning rate can adjust
        during the ramp-up phase of regularization.
    
    entropy_on_classes : bool, default=False
        If True, entropy regularization is grouped by class label and averaged per class.
    
    early_stopping_patience : int, optional
        Number of consecutive epochs with no improvement in validation loss
        before stopping training early.
        If ``None``, this is set automatically based on the warm-up length:
        ``max(1000, theta_warmup // 10)``.
        This is typically longer than ``scheduler_patience`` to give the model
        enough time to adapt, but still prevent wasted computation if the loss
        plateaus.
    
    baseline_rmse : float or None, default=None
        Precomputed baseline RMSE for normalization. If None and `compute_baseline=True`, 
        it is computed automatically.
    
    baseline_zero_reg : float or None, default=None
        Precomputed baseline zero regularization value for normalization.
    
    baseline_entropy_reg : float or None, default=None
        Precomputed baseline entropy regularization value for normalization.
    
    compute_baseline : bool, default=False
        If True, fits an unregularized baseline model to compute normalization constants.
    """
    
    def __init__(
        self,
        theta=0.5,
        epsilon=0.5,
        w_init=None,
        l1_ratio=0.5,
        device='cpu',
        verbose=False,
        min_delta=1e-5,
        n_components=2,
        random_state=42,
        max_iter=int(1e5),
        theta_warmup=None,
        non_negative=False,
        learning_rate=1e-1,
        feature_classes=None,
        scheduler_factor=0.5,
        activation="softplus",
        entropy_scaling="log",
        scheduler_patience=None,
        entropy_on_classes=False,
        zero_warmup=None,
        entropy_warmup=None,
        early_stopping_patience=None,
        baseline_rmse=None,
        baseline_zero_reg=None,
        baseline_entropy_reg=None,
        compute_baseline=False,
        
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
        self.entropy_on_classes = bool(entropy_on_classes)
        self.scheduler_factor = np.clip(scheduler_factor, 0.0, 1.0)
        self.theta_warmup=int(theta_warmup) if theta_warmup is not None else None
        self.scheduler_patience = int(scheduler_patience) if scheduler_patience is not None else None
        self.early_stopping_patience = int(early_stopping_patience) if early_stopping_patience is not None else None
        
        # --- NEW: baselines + flag ---
        self.baseline_rmse = None if baseline_rmse is None else float(baseline_rmse)
        self.baseline_zero_reg = None if baseline_zero_reg is None else float(baseline_zero_reg)
        self.baseline_entropy_reg = None if baseline_entropy_reg is None else float(baseline_entropy_reg)
        self.compute_baseline = bool(compute_baseline)
        # Warmup schedules: "linear", "bezier_convex", "bezier_concave"
        self.zero_warmup = zero_warmup if zero_warmup is not None else "linear"
        self.entropy_warmup = entropy_warmup if entropy_warmup is not None else "bezier_convex"

        self.rmse_list = [] # for plotting/logging
        self.zero_reg_list = [] # for plotting/logging
        self.entr_reg_list = [] # for plotting/logging
        
    def _compute_zero_reg(self, W: torch.Tensor) -> torch.Tensor:
        # --- Zero Regularization ---
        mask = self.encoder.mask_zero_entries
        if not mask.any():
            return W.new_tensor(0.0)
        # per-column counts (>=0), avoid divide-by-zero by masking later
        counts = mask.sum(dim=0)  # shape: (k,)
        active = counts > 0
    
        l1_col_mean = ((W.abs() * mask).sum(dim=0) / counts.clamp_min(1))[active]
        l2_col_mean = (((W ** 2) * mask).sum(dim=0) / counts.clamp_min(1))[active]
    
        if l1_col_mean.numel() == 0:
            return W.new_tensor(0.0)
    
        zero_reg_l1 = l1_col_mean.mean()
        zero_reg_l2 = torch.sqrt(l2_col_mean.mean())
        return self.l1_ratio * zero_reg_l1 + (1 - self.l1_ratio) * zero_reg_l2
        
    def _compute_entropy_reg(self, W: torch.Tensor) -> torch.Tensor:
        # --- Entropy Regularization ---
        # Compute per-feature probs (row-normalized abs weights)
        probs_all = F.normalize(W.abs().clamp_min(1e-8), p=1, dim=1)
        entropies_all = -(probs_all * probs_all.log()).sum(dim=1)
        # Pull class vector (on same device); values: -2 (nan), -1, >=0
        c_full = self.encoder.feature_classes_full  # shape: (d,)
        # Features explicitly marked for entropy: c_i == -1
        unknown_mask = (c_full == -1)
        # Features with known class labels: c_i >= 0
        valid_mask = (c_full >= 0)
        entropy_reg = torch.tensor(0.0, device=W.device)
        entropy_flag = False
        # Always include "-1" features in entropy (unsupervised selectivity)
        if unknown_mask.any():
            unk_mean = entropies_all[unknown_mask].mean()
            entropy_reg += unk_mean
            entropy_flag = True
        # Optionally include class-averaged entropy for labeled features
        if self.entropy_on_classes and valid_mask.any():
            valid_classes = c_full[valid_mask]
            valid_entropies = entropies_all[valid_mask]
            # classes are non-negative ints; compute per-class mean then global mean
            max_cls = int(valid_classes.max().item())
            group_sums = torch.zeros(max_cls + 1, device=W.device)
            group_counts = torch.zeros_like(group_sums)
            group_sums.index_add_(0, valid_classes, valid_entropies)
            group_counts.index_add_(0, valid_classes, torch.ones_like(valid_entropies))
            group_means = group_sums / group_counts.clamp_min(1)
            cls_mean = group_means[group_counts > 0].mean()
            entropy_reg = entropy_reg + cls_mean
            entropy_flag = True
        # If both parts contributed, average them (matches your intent)
        if self.entropy_on_classes and unknown_mask.any() and valid_mask.any():
            entropy_reg = entropy_reg / 2.0
        # Scale (e.g., 1/log k); self.entropy_scaling_multiplier set in fit()
        entropy_reg = entropy_reg * self.entropy_scaling_multiplier
        # Store a flag for combining with epsilon later
        self.entropy_flag = 1 if entropy_flag else 0
        return entropy_reg

    def _sched(self, kind: str, u: torch.Tensor) -> torch.Tensor:
        # u in [0,1]
        if kind == "linear":
            return u
        elif kind == "bezier_convex":
            # (0,0)-(1,0)-(1,1): slow start, fast finish; convex
            return (1.0 - torch.sqrt(1.0 - u))**2
        elif kind == "bezier_concave":
            # mirror of bezier_convex over y=x: fast start, slow finish; concave
            # exact inverse of the convex curve: s(u) = 2*sqrt(u) - u
            return 2.0 * torch.sqrt(u) - u
        else:
            raise ValueError(f"Unknown warmup kind: {kind!r}")
    
    def _compute_reg_weights(self, epoch: int):
        """
        Returns (lambda_zero, lambda_ent, lambda_tot) for this epoch,
        using self.zero_warmup and self.entropy_warmup.
        Uses a fixed epsilon split regardless of which regularizers are active.
        """
        u = torch.tensor((epoch + 1) / self.theta_warmup, device=self.device).clamp(0.0, 1.0)
        s_zero = self._sched(self.zero_warmup, u)
        s_ent  = self._sched(self.entropy_warmup, u)
    
        lambda_zero = self.theta * (1.0 - self.epsilon) * s_zero
        lambda_ent  = self.theta * self.epsilon * s_ent
        lambda_tot  = lambda_zero + lambda_ent
        return lambda_zero, lambda_ent, lambda_tot

    def _compute_baselines_if_needed(self, X_tensor, X_val_tensor, eps0: float):
        """
        Encapsulates the baseline computation block (no behavior changes).
        Sets: self.baseline_rmse, self.baseline_zero_reg, self.baseline_entropy_reg.
        """
        if (self.baseline_rmse is not None) and (self.baseline_zero_reg is not None) and (self.baseline_entropy_reg is not None):
            return
    
        if not self.compute_baseline:
            # default neutral dividers
            self.baseline_rmse = 1.0
            self.baseline_zero_reg = 1.0
            self.baseline_entropy_reg = 1.0
            return
    
        base = CLEAR(
            theta=0.0,
            epsilon=0.0,
            l1_ratio=self.l1_ratio,
            device=self.device,
            w_init=self.w_init,
            verbose=0,
            min_delta=self.min_delta,
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            non_negative=self.non_negative,
            learning_rate=self.learning_rate,
            feature_classes=self.feature_classes,  # keep masks for evaluating regs
            scheduler_factor=self.scheduler_factor,
            activation=self.activation,
            entropy_scaling=self.entropy_scaling,
            scheduler_patience=self.scheduler_patience,
            entropy_on_classes=self.entropy_on_classes,
            early_stopping_patience=self.early_stopping_patience,
            # do NOT recurse baseline computation
            baseline_rmse=1.0,
            baseline_zero_reg=1.0,
            baseline_entropy_reg=1.0,
            compute_baseline=False,
        )
        base.fit(X_tensor.detach().cpu().numpy(), X_val=None if X_val_tensor is None else X_val_tensor.detach().cpu().numpy())
        with torch.no_grad():
            Xb = X_val_tensor if X_val_tensor is not None else X_tensor
            Xb_hat = base.model(Xb)
            baseline_rmse = torch.sqrt(self.loss_fn(Xb_hat, Xb)).item()
            baseline_W = base.encoder.activation(base.encoder.W_raw)
            baseline_zero_reg = base._compute_zero_reg(baseline_W).item()
            baseline_entropy_reg = base._compute_entropy_reg(baseline_W).item()
    
        self.baseline_rmse = max(baseline_rmse, eps0)
        self.baseline_zero_reg = max(baseline_zero_reg, eps0)
        self.baseline_entropy_reg = max(baseline_entropy_reg, eps0)
    
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
        self : CLEAR
            The fitted model.
        """
        # Training setup
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device) if X_val is not None else None
        n_features = X_tensor.shape[1]
    
        # Interpret special feature_classes shorthand
        fc = self.feature_classes
        if isinstance(fc, (int, float)):
            if fc == -1:
                self.feature_classes_ = np.full((n_features,), -1, dtype=np.float32)
            elif fc == -2 or (isinstance(fc, float) and np.isnan(fc)):
                # treat -2 or NaN scalar as "no regularization for all features"
                self.feature_classes_ = None
            else:
                raise ValueError(
                    f"Unsupported scalar feature_classes={fc}. "
                    "Use -1 for entropy-only on all features, or -2/NaN for none."
                )
        elif isinstance(fc, str):
            s = fc.strip().lower()
            if s in {"-1"}:
                self.feature_classes_ = np.full((n_features,), -1, dtype=np.float32)
            elif s in {"-2", "nan", "none", "null"}:
                self.feature_classes_ = None
            else:
                raise ValueError(f"Unsupported feature_classes string: {fc}")
        elif isinstance(fc, (list, tuple, np.ndarray)):
            arr = np.asarray(fc, dtype=np.float32)
            if arr.shape[0] != n_features:
                raise ValueError(
                    f"feature_classes has length {arr.shape[0]} but expected {n_features}"
                )
            self.feature_classes_ = arr
        else:
            # Default: no regularization metadata provided
            self.feature_classes_ = None

        # theta warmup heuristic 
        if self.theta_warmup is None:
            n, d = X.shape
            self.theta_warmup = int(np.sqrt(n * d))
        else:
            self.theta_warmup = int(self.theta_warmup)
        
        # scheduler patience heuristic: 1/50 of warmup, min 200
        if self.scheduler_patience is None:
            self.scheduler_patience = max(200, int(self.theta_warmup / 50))
        else:
            self.scheduler_patience = int(self.scheduler_patience)
        
        # early stopping patience heuristic: 1/10 of warmup, min 1000
        if self.early_stopping_patience is None:
            self.early_stopping_patience = max(1000, int(self.theta_warmup / 10))
        else:
            self.early_stopping_patience = int(self.early_stopping_patience)
        
        # Build model using parsed feature_classes
        self._build_model(X_tensor.shape[1])
        # --- initialize entropy_flag before training ---
        fc_full = self.encoder.feature_classes_full
        unknown_mask = (fc_full == -1)         # always included in entropy
        valid_mask   = (fc_full >= 0)          # used if entropy_on_classes=True
        self.entropy_flag = bool(unknown_mask.any() or (self.entropy_on_classes and valid_mask.any()))

        # Scaling factor for entropy
        if self.entropy_scaling is None:
            self.entropy_scaling_multiplier = 1.0
        elif self.entropy_scaling == "log":
            self.entropy_scaling_multiplier = 1.0 if self.n_components <= 1 else 1.0 / np.log(self.n_components)
        elif self.entropy_scaling == "exp":
            self.entropy_scaling_multiplier = 1.0 / np.exp(self.n_components)
        else:
            self.entropy_scaling_multiplier = float(self.entropy_scaling)

        # --- NEW: compute or set baselines (per config) ---
        eps0 = 1e-12
        self._compute_baselines_if_needed(X_tensor, X_val_tensor, eps0)

        best_val = float('inf')
        epochs_no_improve = 0
        sched_epochs_no_improve = 0
        
        for epoch in range(self.max_iter):
            self.model.train()
            self.optimizer.zero_grad()
        
            # Forward
            X_hat = self.model(X_tensor)
            rmse = torch.sqrt(self.loss_fn(X_hat, X_tensor))
            n_rmse = rmse / max(self.baseline_rmse, eps0)
        
            # --- Regularization terms (compute & normalize FIRST) ---
            W = self.encoder.activation(self.encoder.W_raw)
            zero_reg = self._compute_zero_reg(W)
            entropy_reg = self._compute_entropy_reg(W)
            n_zero_reg = zero_reg / max(self.baseline_zero_reg, eps0)
            n_entropy_reg = entropy_reg / max(self.baseline_entropy_reg, eps0)
        
            # --- Weights (linear zero, chosen warmup for entropy; epsilon only if both exist) ---
            lambda_zero, lambda_ent, lambda_tot = self._compute_reg_weights(epoch)
        
            # Weighted reg and final loss
            reg_total_weighted = lambda_zero * n_zero_reg + lambda_ent * n_entropy_reg
            loss = (1.0 - lambda_tot) * n_rmse + reg_total_weighted

            self.rmse_list.append(rmse.item())
            self.zero_reg_list.append(zero_reg.item())
            self.entr_reg_list.append(entropy_reg.item())
            
            loss.backward()
            self.optimizer.step()

            # Validation & Logging
            val_rmse, val_loss = rmse, loss
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_hat = self.model(X_val_tensor)
                    val_rmse = torch.sqrt(self.loss_fn(X_val_hat, X_val_tensor))
                    n_val_rmse = val_rmse / max(self.baseline_rmse, eps0)
                    val_loss = (1 - theta_t) * n_val_rmse + theta_t * reg_total

            if self.verbose and (epoch % self.verbose == 0) and (epoch != 0):
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:5d} | NoImp: {epochs_no_improve:3d} | "
                    f"Train Loss: {loss.item():.5f} | Train Error: {rmse.item():.5f} | "
                    f"Val Loss: {val_loss.item():.5f} | Val Error: {val_rmse.item():.5f} | "
                    f"Zero Reg: {zero_reg.item():.5f} | Entropy Reg: {entropy_reg.item():.5f} | LR: {lr:.5f}",
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

        self.rmse_ = rmse
        self.zero_reg_ = zero_reg
        self.entropy_reg_ = entropy_reg
        
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
            If True, each row is normalized to unit L1 norm (sum to 1).

        Returns
        -------
        W : ndarray of shape (n_features, n_components)
            The weight matrix after activation and optional scaling.
        """
        W_tensor = self.encoder.activation(self.encoder.W_raw)
        if apply_scaling:
            W_tensor = F.normalize(W_tensor, p=1, dim=1)
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
        self : CLEAR
            The model moved to the target device.
        """
        self.device = device
        if hasattr(self, "model"):
            self.model.to(device)
        return self
