# Grouped Autoencoder

A **linear autoencoder** built with PyTorch, designed for dimensionality reduction with support for **group-aware feature constraints**, **entropy-based regularization**, and **non-negativity**. Compatible with scikit-learn pipelines.

## 🔍 Overview

This autoencoder introduces **structured constraints** on the encoder weights (`W`), informed by user-supplied feature groupings (called `feature_classes`). It allows:

- Forcing certain features to only load onto specific latent components.
- Applying entropy-based regularization for ungrouped (NaN-labeled) features.
- Mixing structure loss and reconstruction via a tunable parameter `theta`.

## ✨ Features

- ✅ Group-aware encoder constraints via `feature_classes`
- 🔐 Optional **non-negative encoding weights**
- 🧠 Entropy-based regularization for ungrouped features
- 🔄 Compatible with `scikit-learn` transformers
- 🛑 Early stopping and learning rate scheduling
- 🚀 GPU support via `.to('cuda')`

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🧠 Usage

```python
from grouped_autoencoder import GroupedAutoencoder
import numpy as np

# Input data
X = np.random.rand(100, 10)

# Optional: feature grouping (e.g., feature_classes[i] = 0 means feature i belongs to group 0)
feature_classes = np.array([0, 0, 1, 1, np.nan, 2, 2, np.nan, 0, 1])

# Initialize model
model = GroupedAutoencoder(
    n_components=3,
    feature_classes=feature_classes,
    theta=0.9,              # Higher = stronger regularization
    non_negative=True,
    verbose=True
)

# Fit model
model.fit(X)

# Low-dimensional embedding
Z = model.transform(X)

# Reconstructed input
X_recon = model.inverse_transform(Z)

# Encoder weight matrix
W = model.get_W()
```

## 📊 Parameters

| Name                  | Description |
|-----------------------|-------------|
| `n_components`        | Latent dimension size |
| `feature_classes`     | Array of size (n_features,) with group labels or `np.nan` |
| `theta`               | Mix between reconstruction loss (0.0) and structure regularization (1.0) |
| `non_negative`        | Constrain encoder weights to be non-negative |
| `max_iter`            | Max training iterations |
| `early_stopping_patience` | Stop if no improvement after this many epochs |
| `random_state`        | Reproducibility |

## 📌 Notes

- When `feature_classes` is provided:
  - Each non-NaN entry `i` maps feature `i` to a latent component `c`.
  - For such features, the encoder is regularized to **only load** onto their assigned component.
  - Features with `NaN` are not structurally constrained, but **encouraged to be sparse** via entropy minimization.
- `theta=0.0` means **pure autoencoder**.
- `theta=1.0` (internally clipped to `1 - 1e-6`) means **purely regularized projection**.

## 📈 Tips

- Use `model.to('cuda')` to leverage GPU training.
- Use high `theta` if your feature classes are trustworthy.

## 🛠 Dependencies

- `torch`
- `numpy`
- `scikit-learn`

## 📄 License

GPL-3.0


