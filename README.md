# 🧠 Grouped Autoencoder

A **PyTorch-based linear autoencoder** framework designed for **interpretable dimensionality reduction** under structured constraints. This model supports **group-aware regularization**, **non-negativity**, **entropy-based sparsity control**, and is **scikit-learn-compatible**.

---

## 🔬 Scientific Motivation

Conventional autoencoders are optimized for reconstruction fidelity without regard to **interpretability** or **feature structure**. The Grouped Autoencoder (GAE) augments this by imposing **structured priors** over the encoder weights:

- Features can be **assigned to latent components**, mimicking group-wise loadings found in biological, economic, or social data.
- **Entropy minimization** encourages **sparse**, **localized** representations (low-overlap features).
- **Zero suppression** penalizes off-component loadings to enforce group-exclusive mappings.
- Optionally constrains weights to be **non-negative**, promoting parts-based learning à la NMF.

This makes GAE suitable for applications in:
- Genomics (pathway-constrained embeddings)
- NLP (topic-aligned latent factors)
- Recommender systems (user-group decomposition)
- Explainable ML

---

## ✨ Features

✅ Group-aware weight constraints (`feature_classes`)  
📉 Entropy regularization (feature-level or class-level)  
⚖️ L1/L2 structural regularization (`l1_ratio`)  
🔐 Optional non-negative weights via `softplus` or `sigmoid`  
📈 Early stopping, LR scheduling  
🧱 Sklearn-compatible interface (`fit`, `transform`, `predict`)  
🚀 GPU acceleration via `.to('cuda')`  

---

## 📦 Installation

```bash
pip install torch numpy scikit-learn
```

---

## 🧪 Example Usage

```python
from grouped_autoencoder import GroupedAutoencoder
import numpy as np

# Sample input (n_samples x n_features)
X = np.random.rand(100, 10)

# Grouping constraints
feature_classes = np.array([0, 0, 1, 1, -1, 2, 2, np.nan, 0, 1])

model = GroupedAutoencoder(
    n_components=3,
    feature_classes=feature_classes,
    theta=0.95,
    entropy_on_classes=True,
    activation="softplus",
    non_negative=True,
    verbose=100,
)

model.fit(X)
Z = model.transform(X)
X_recon = model.predict(X)
W = model.get_W(apply_scaling=True)
```

---

## ⚙️ Parameters

| Parameter               | Description |
|-------------------------|-------------|
| `n_components`          | Size of latent space. |
| `feature_classes`       | Array (n_features,) indicating group constraints:<br>• `>= 0`: apply zero + entropy reg for that component<br>• `-1`: entropy-only<br>• `np.nan`: no regularization |
| `theta` (default=1.0)   | Balance between MSE and regularization. |
| `epsilon` (default=0.5) | Tradeoff between zero vs. entropy reg. |
| `l1_ratio` (default=0.5)| L1 vs. L2 mix in zero reg. |
| `activation`            | Used for non-negative weights: `'sigmoid'` or `'softplus'`. |
| `non_negative`          | Enforces non-negativity on encoder weights. |
| `entropy_on_classes`    | If True, computes average entropy per class. |
| `entropy_scaling`       | One of `'log'`, `'exp'`, or `None`. |
| `learning_rate`         | Initial LR for Adam optimizer. |
| `early_stopping_patience` | Stop if no improvement for N epochs. |
| `scheduler_patience`    | LR decay trigger window. |
| `scheduler_factor`      | LR decay factor. |
| `verbose`               | If int > 0, logs every N epochs. |
| `random_state`          | Reproducibility seed. |
| `device`                | `'cpu'` or `'cuda'`. |

---

## 🧠 Regularization Details

### 🔹 Zero Regularization (Structure Loss)
Encourages features to load **only** on their assigned component by penalizing other entries in the weight matrix.

Uses a mixture of:
- **L1 norm**: promotes hard sparsity
- **L2 norm**: promotes soft exclusivity

### 🔹 Entropy Regularization (Sparsity Loss)
Promotes **low entropy** (peaky distributions) across component weights for each feature.

- If `entropy_on_classes=False`: applied per feature
- If `entropy_on_classes=True`: averaged per class

### 🔹 Combined Loss
The total loss is a mixture:

```
total_loss = (1 - θ) * RMSE + θ * [(1 - ε) * zero_reg + ε * entropy_reg]
```

where:
- θ = regularization weight (`theta`)
- ε = entropy/zero balance (`epsilon`)

---

## 📊 Understanding the Weights

Calling `model.get_W(apply_scaling=True)` returns the encoder matrix `W` with optional L1 normalization (columns sum to 1).

- Rows = input features
- Columns = latent components
- Values = learned loading strengths

Perfect for interpretability and plotting (e.g., heatmaps).

---

## 🏁 Tips for Use

- Set `theta` near **1.0** for projection-driven embeddings.
- Use `'softplus'` if gradients are unstable with `'sigmoid'`.
- If feature grouping is noisy, use a lower `epsilon` to emphasize zero reg.
- Use `model.to('cuda')` **before** calling `.fit(X)` for GPU acceleration.


---

## 🎯 Applications

The Grouped Autoencoder is designed for tasks requiring **interpretable representations** that reflect known or hypothesized structure in the data. Applications span across scientific and non-scientific domains.

### 🔬 STEM Applications

- **Genomics & Bioinformatics**: Map genes to known pathways or functional clusters by assigning feature_classes based on known gene ontology.
- **Neuroscience**: Decompose brain signals (EEG, fMRI) with anatomical or functional priors, enabling structured latent representations.
- **Healthcare Informatics**: Project patient feature data (e.g., symptoms, labs) into disease-oriented embeddings.
- **Recommender Systems**: Encourage items (features) to associate with a subset of user-interest categories.

### 💬 Non-STEM / Social Science Applications

- **Survey Analysis & Psychometrics**: When dealing with high-dimensional survey responses (e.g., personality tests, social attitudes), setting `feature_classes = -1` allows the model to discover sparse and interpretable latent dimensions (e.g., behavioral archetypes) via entropy-based regularization.
- **Sociology / Education**: Interpret latent traits in structured assessments by aligning questionnaire items with hypothesized constructs.
- **Marketing**: Discover underlying customer segments by analyzing question-level feedback data (e.g., product surveys).
- **Political Science**: Identify ideological dimensions from question-level voting or polling data, even without clear prior groupings.

### 🧠 Exploratory Research

If no clear structure is known, using `feature_classes = -1` or `np.nan` allows the model to **learn sparse representations** in an unsupervised way—ideal for exploratory latent factor analysis or pretraining for downstream models.

---

---

## 📁 File Structure

```
grouped_autoencoder/
├── grouped_autoencoder.py      # Main model class (Encoder, Decoder, GAE)
└── README.md                   # You're here.
```

---

## 📄 License

GLP-3.0
