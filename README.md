# r-hash-SANE: Weight-Space Learning for NeRFs

> **Neural Radiance Fields meet Structured Affine Neural Embeddings.**  
> A transformer-based autoencoder for learning 3D scene representations directly in the **weight space** of Instant-NGP models.

---

## 🚀 Overview

This project extends the **Structured Affine Neural Embeddings (SANE)** algorithm to **Instant Neural Graphics Primitives (Instant-NGP)** — bringing weight-space learning into the domain of 3D scene representations.

Instead of analyzing or generating images, we analyze the **weights** of trained Instant-NGP networks to learn disentangled latent embeddings that encode **object identity** and **rotational transformations**.  

We design a transformer-based autoencoder that:
- Tokenizes both **hash grid features** and **MLP weights** from Instant-NGP.
- Learns **rotation-aware** and **object-specific** latent representations.
- Combines **reconstruction**, **contrastive**, and **rotation-prediction** objectives.

---

## 🧠 Key Ideas

- **Weight-space learning** instead of data-space learning:  
  Neural representations are treated as data points in parameter space.

- **Hybrid tokenization strategy**:  
  - Hash layers → spatially structured 256D tokens sampled from 3D space.  
  - MLP layers → grouped neuron blocks mapped to 256D tokens.

- **Rotation-aware positional encoding**:  
  Custom positional embeddings preserve 3D structure and Instant-NGP layer organization.

- **Multi-objective training**:
  ```
  L = 0.4 * L_recon + 0.05 * L_contrastive + 0.53 * L_rotation + 0.02 * ||Z||²
  ```

---

## 📦 Architecture

A **transformer autoencoder** forms the core of the system:

```
Instant-NGP Weights
        ↓
Tokenization (Hash + MLP + POS)
        ↓
Encoder (Transformer)
        ↓
Latent Representation Z
        ↓
Decoder (Reconstruction)
  ├─ NT-Xent Head (Contrastive)
  └─ Rotation Head (Quaternion Prediction)
```

- **Encoder/Decoder:** 4-layer transformer, 4 attention heads, 1024-d embeddings.  
- **Latent space:** 128-dim per token, split into “content” and “rotation” subspaces.  
- **[POS] tokens:** Dedicated 10 tokens for predicting rotation quaternions.  

---

## 📊 Dataset

- **342 3D objects**, each trained as **Instant-NGP** under 6 rotational viewpoints.  
- Each sample includes:
  - 16 **hash tables** (~6.5M params)
  - 6 **MLP layers** (~20K params)
- Total input sequence per model: **443 tokens × 256 dims**.  

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Batch size | 32 |
| Epochs | 500 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Dropout | 0.3–0.4 |
| Contrastive temp. | 0.1 |
| Scheduler | OneCycleLR |

---

## 🧩 Results Summary

| Task | Outcome |
|------|----------|
| **Reconstruction** | Converged (~0.1 loss) — structure preserved. |
| **Contrastive (NT-Xent)** | Successful rotation-invariant embeddings (<0.1 loss). |
| **Rotation Prediction** | Overfitting; generalization limited — data too sparse. |

Key insight: The model can *understand rotation-invariant structure* in NeRF weight space, but **true disentanglement of rotation vs. content** remains difficult with limited discrete rotations.

---

## 🧭 Future Work

- Larger and continuous-rotation datasets.  
- Improved positional encodings (concatenative, not additive).  
- Exploring **continuous rotation interpolation** in latent space.  
- Rebalancing latent capacity between `[POS]` and content tokens.  

---

## 🧑‍💻 Authors

- **Zilong Liu** – Geoinformation Science  
- **Roman Oelfken** – Information Systems  
- **Berk Can Özmen** – Computer Science *(repo maintainer)*  
- **Can-Philipp Tura** – Computer Science  

Technische Universität Berlin  

---

## 📚 References

- Schürholt et al., *Towards Scalable and Versatile Weight Space Learning*, arXiv:2406.09997  
- Müller et al., *Instant Neural Graphics Primitives*, ACM TOG, 2022  
- Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields*, CACM, 2021  

---

## 🏗️ Repo Structure (expected)

```
r-hash-SANE/
│
├── src/
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── ntx_head.py
│   │   └── rotation_head.py
│   ├── data/
│   │   ├── dataset_loader.py
│   │   └── tokenization.py
│   └── train.py
│
├── configs/
│   └── hyperparams.yaml
│
├── results/
│   ├── logs/
│   ├── reconstructions/
│   └── checkpoints/
│
├── README.md
└── requirements.txt
```
