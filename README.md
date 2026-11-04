## Codebase for the paper "Enhancing Assessment of Self-Consistency in LLM Explanations using Perturbation Strength"

Phuong Quynh Le, Kemal Kurniawan, and Jey Han Lau. 2026. Enhancing Assessment of Self-Consistency in LLM
Explanations using Perturbation Strength. [Preprint](https://doi.org/10.48550/arXiv.2609.30849), arXiv:2609.30849. Under review at ALTA 2026.

**Note**: This repository is a work in progress. Some code is not yet available and the repository structure is still being organised. Further updates will be made to improve code quality and completeness. Please refer to the paper cited above for the complete report. 
- The annotated data can be found in [annotation_data](https://github.com/ple30/enhance-self-consistency-check/tree/main/annotation_data)

### MLP on frozen sentence embeddings (cross-validated)
`train_cv_mlp.py` embeds every text once with a frozen sentence-transformer (cached in `results/emb_cache/`), builds pair features (`|u−v|`, `u⊙v`, cosine by default; optionally `u`, `v`) plus a domain one-hot, and trains a small MLP. It reads all labelled pairs from `classifier_train_data/train.csv` and runs grouped, stratified k-fold CV (pairs with the same original text never straddle folds; a given seed always gives the same folds), reporting linear-weighted kappa (primary), Pearson correlation, accuracy, macro-F1 and MAE, overall and per domain. `--classifier logreg` replaces the MLP with scikit-learn's logistic regression, with its regularisation strength `C` chosen in each training fold by an inner grouped CV.

```bash
pip install torch sentence-transformers scikit-learn pandas
python train_cv_mlp.py --embedding-model all-mpnet-base-v2 --seeds 42 43 44 45 46 --final-fit
python train_cv_mlp.py --classifier logreg --seeds 42 43 44 45 46   # logistic-regression baseline
python train_cv_mlp.py --features u,v,diff,prod,cos --pca 32 --swap-aug
```

### Evaluating on the test set
`evaluate_mlp.py` loads a final model from `train_cv_mlp.py --final-fit` (MLP or logistic regression), embeds the pairs in `annotation_data/test.csv` with the same embedding model and features, and reports linear-weighted kappa and Pearson correlation (plus accuracy, macro-F1, MAE) overall, per domain and per source file, with a confusion matrix. Test pairs whose original text also appears in the training data are flagged, and metrics are also reported without them.

```bash
python train_cv_mlp.py --embedding-model all-mpnet-base-v2 --seeds 42 43 44 45 46 --final-fit --output-dir results/mlp_mpnet
python evaluate_mlp.py --model-path results/mlp_mpnet/final_model.joblib
```
