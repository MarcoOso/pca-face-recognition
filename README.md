# PCA + CRC Face Recognition

Dimensionality reduction (PCA / eigenfaces) applied to face classification on the
[Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset, using a
from-scratch Collaborative Representation Classifier (CRC). The goal is to measure
how much PCA helps (or hurts) both accuracy and runtime compared to running CRC
directly on raw pixel features.

Adapted from a team project for MAP 4112 (with Sparsh Pandey — see [main.tex](main.tex)
for the full write-up). This repo focuses on my contribution: the PCA and CRC
implementation and the experiments below.

## Approach

1. **Data**: LFW, keeping only identities with ≥50 images, images resized to 40% and
   flattened to 1,850-dimensional grayscale vectors.
2. **Split**: stratified 60% train / 15% val / 25% test (fixed seed).
3. **Preprocessing**: standardize features using train-set statistics, then
   ℓ2-normalize each row.
4. **PCA (via SVD)**: fit on centered train data; the number of components *k* is
   chosen from a target cumulative explained-variance ratio.
5. **CRC**: implemented from scratch — precompute `P = (DᵀD + λI)⁻¹Dᵀ` from the
   training dictionary `D`, then classify each sample by whichever class's
   reconstruction has the lowest residual.
6. **Tuning**: grid search over `λ` (and, for the PCA variant, the variance ratio
   `{0.90, 0.95, 0.99}`), selecting by validation accuracy, retraining on train+val,
   and evaluating once on the held-out test set.

## Results

| Model        | Test Accuracy | Feature Dim |
|--------------|---------------|-------------|
| CRC, no PCA  | 48.21%        | 1,850       |
| CRC + PCA    | **50.51%**    | 373 (k at 99% variance retained) |

Variance ratio → number of components (*k*):

| Variance retained | k   |
|--------------------|-----|
| 90%                | 81  |
| 95%                | 154 |
| 99%                | 373 |

Total runtime across all hyperparameter configurations: **7.6s** for CRC without PCA
(5 configs) vs. **3.8s** for CRC with PCA (15 configs) — PCA searched a larger grid
in about half the time.

**Takeaway**: dropping from 1,850 to 373 dimensions (retaining 99% of variance) both
sped up training and slightly improved test accuracy, likely by filtering out
low-variance pixel noise that CRC would otherwise overfit to.

![CRC hyperparameter comparison](crc_comparison_table.png)
![Confusion matrices: CRC with and without PCA](confusion_matrices.png)

## Repo contents

- [`pca_crc_face_recognition.ipynb`](pca_crc_face_recognition.ipynb) — full
  implementation and experiments (PCA, CRC, tuning, plots)
- [`main.tex`](main.tex) — project write-up (introduction, discussion, results,
  conclusion)
- `crc_comparison_table.png`, `confusion_matrices.png` — result figures

## Setup

```bash
pip install -r requirements.txt
jupyter notebook pca_crc_face_recognition.ipynb
```

The notebook downloads LFW automatically via `sklearn.datasets.fetch_lfw_people`
on first run (cached locally by scikit-learn afterward).

## Limitations

PCA components are linear and orthogonal, so any useful *nonlinear* structure in
the pixel data isn't captured — this caps how much PCA alone can help. Retaining
too little variance also hurts accuracy, so the variance-ratio threshold matters;
in these experiments, 99% was the best of the three ratios tried.
