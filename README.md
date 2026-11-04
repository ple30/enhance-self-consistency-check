## Codebase for the paper "Enhancing Assessment of Self-Consistency in LLM Explanations using Perturbation Strength"

Phuong Quynh Le, Kemal Kurniawan, and Jey Han Lau. 2026. Enhancing Assessment of Self-Consistency in LLM
Explanations using Perturbation Strength. [Preprint](https://doi.org/10.48550/arXiv.2609.30849), arXiv:2609.30849. Under review at ALTA 2026.

**Note**: This repository is a work in progress. Some code is not yet available and the repository structure is still being organised. Further updates will be made to improve code quality and completeness. Please refer to the paper cited above for the complete report. 
- The annotated data can be found in [annotation_data](https://github.com/ple30/enhance-self-consistency-check/tree/main/annotation_data)

### Perturbation-strength classifier (cross-validated fine-tuning)
`train_cv.py` fine-tunes a cross-encoder that predicts the perturbation strength (0 = no/minimal change ... 3 = strong change) of a `(text_a, text_b)` pair, using the domain (CoT vs. Question) as an extra input (`--domain-mode both|embed|prefix|none`). It reads all labelled pairs from `classifier_train_data/train.csv` (there is no separate validation file) and runs grouped, stratified k-fold CV (rows with the same original text never straddle folds).

```bash
pip install torch transformers sentencepiece scikit-learn pandas openpyxl
python train_cv.py --model microsoft/deberta-v3-small --epochs 10 --seeds 42 43 44 --swap-aug --final-fit
```
Outputs (in `results/cv_<model>_<domain-mode>/`): per-epoch fold metrics, out-of-fold predictions, `summary.json` (linear-weighted Cohen's kappa — the primary metric, used to select the number of epochs — Pearson correlation, accuracy, macro-F1, MAE; overall and per domain), and optionally the final model trained on all data.

### MLP on frozen sentence embeddings (cross-validated)
`train_cv_mlp.py` embeds every text once with a frozen sentence-transformer (cached in `results/emb_cache/`), builds pair features (`|u−v|`, `u⊙v`, cosine by default; optionally `u`, `v`) plus a domain one-hot, and trains a small MLP. It uses the same data, folds, epoch selection and metrics as `train_cv.py`, so results are directly comparable. `--hidden 0` gives a logistic-regression baseline.

```bash
pip install torch sentence-transformers scikit-learn pandas
python train_cv_mlp.py --embedding-model all-mpnet-base-v2 --seeds 42 43 44 45 46 --final-fit
python train_cv_mlp.py --hidden 0                            # logistic-regression baseline
python train_cv_mlp.py --features u,v,diff,prod,cos --pca 32 --swap-aug
```
