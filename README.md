## Codebase for the paper "Enhancing Assessment of Self-Consistency in LLM Explanations using Perturbation Strength"

Phuong Quynh Le, Kemal Kurniawan, and Jey Han Lau. 2026. Enhancing Assessment of Self-Consistency in LLM
Explanations using Perturbation Strength. [Preprint](https://doi.org/10.48550/arXiv.2609.30849), arXiv:2609.30849. Under review at ALTA 2026.

**Note**: This repository is a work in progress. Some code is not yet available and the repository structure is still being organised. Further updates will be made to improve code quality and completeness. Please refer to the paper cited above for the complete report. 
- The annotated data can be found in [annotation_data](https://github.com/ple30/enhance-self-consistency-check/tree/main/annotation_data)

### MLP on frozen sentence embeddings (cross-validated)
`train_cv_mlp.py` embeds every text once with a frozen sentence-transformer (cached in `results/emb_cache/`). Each pair is represented as `[u, v]`, the embeddings of the original and the perturbed text, plus a domain one-hot (CoT vs. Question), and a small MLP predicts the perturbation strength (0 = no/minimal change ... 3 = strong change). Question texts are cut before the word "Options", so only the question itself is embedded (the answer options are identical in the original and perturbed question); `--keep-options` embeds the full text, and `evaluate_mlp.py` applies the same rule as the model was trained with. Because the input depends on the order of the two texts, `--swap-aug` (also train on the reversed pair) is worth trying. It reads all labelled pairs from `classifier_train_data/train.csv` and runs grouped, stratified k-fold CV (pairs with the same original text never straddle folds; a given seed always gives the same folds), reporting linear-weighted kappa (primary), Pearson correlation, accuracy, macro-F1 and MAE, overall and per domain.

```bash
pip install torch sentence-transformers scikit-learn pandas
python train_cv_mlp.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --hidden 32 --swap-aug --seeds 42 43 44 45 46 --final-fit --output-dir results/mlp_minilm
python train_cv_mlp.py --pca 32 --hidden 16 --seeds 42 43 44 45 46   # reduce [u, v] to 32 components first
```

### Ordinal logistic regression on frozen sentence embeddings (nested CV)
`train_cv_ordinal.py` is the linear, ordinal counterpart of the MLP: a proportional-odds (cumulative-logit) logistic regression with an L2 penalty, which respects the order 0 < 1 < 2 < 3 and has one weight per feature. It uses the same embeddings (and cache), Question-text cutting, folds and metrics as `train_cv_mlp.py`, so results with the same `--seeds` are directly comparable. By default a pair is represented by the symmetric features `[|u - v|, u * v]` plus `cos(u, v)` and a domain indicator; `--pair-mode` also offers `diff`, `prod`, `concat` (`[u, v]`, as in the MLP), `full`, and `none` (with the cosine: a cosine-only baseline). `--pca N` reduces the embedding features first (fitted per fold).

The penalty `alpha` is chosen by nested CV: an inner grouped, stratified CV on each outer training fold picks it, so the reported out-of-fold scores are not tuned on the folds they are measured on. `--select-metric log_loss` is usually a more stable choice than kappa at this sample size. `alpha_curve.csv` lists the outer-CV score of every alpha for inspection only. `--final-fit` saves a model that `evaluate_mlp.py` evaluates on the annotation files (written to an `ordinal` column by default).

```bash
pip install scipy scikit-learn pandas sentence-transformers
python train_cv_ordinal.py --seeds 42 43 44 45 46 --final-fit --output-dir results/ordinal
python train_cv_ordinal.py --pair-mode diffprod --pca 16 --select-metric log_loss --seeds 42 43 44 45 46
python train_cv_ordinal.py --pair-mode none --seeds 42 43 44 45 46   # cosine-only baseline
python evaluate_mlp.py --model-path results/ordinal/final_model.joblib
```

### Evaluating on the test set
`evaluate_mlp.py` loads a final model from `train_cv_mlp.py --final-fit` (or `train_cv_ordinal.py --final-fit`) and reads the annotation Excel files in `annotation_data` directly (every file, every sheet). It predicts every text pair with the same embedding model, including pairs without a human score, and computes linear-weighted kappa and Pearson correlation (plus accuracy, macro-F1, MAE) on the scored pairs, after the same cleaning as the training data: overall, per domain and per source file, with a confusion matrix. Scored pairs whose original text also appears in the training data are flagged, and metrics are also reported without them.

The predictions are written to a column `mlp` (`ordinal` for ordinal models) in the Excel files (reused if it exists, otherwise added after the last column). Files are saved in place; use `--excel-out-dir DIR` to write copies instead, `--excel-column NAME` for another column name, or `--no-write` to only compute metrics.

```bash
python evaluate_mlp.py --model-path results/mlp_minilm/final_model.joblib
```
