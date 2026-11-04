## Codebase for the paper "Enhancing Assessment of Self-Consistency in LLM Explanations using Perturbation Strength"

Phuong Quynh Le, Kemal Kurniawan, and Jey Han Lau. 2026. Enhancing Assessment of Self-Consistency in LLM
Explanations using Perturbation Strength. [Preprint](https://doi.org/10.48550/arXiv.2609.30849), arXiv:2609.30849. Under review at ALTA 2026.

**Note**: This repository is a work in progress. Some code is not yet available and the repository structure is still being organised. Further updates will be made to improve code quality and completeness. Please refer to the paper cited above for the complete report. 
- The annotated data can be found in [annotation_data](https://github.com/ple30/enhance-self-consistency-check/tree/main/annotation_data)

### MLP on frozen sentence embeddings (cross-validated)
`train_cv_mlp.py` embeds every text once with a frozen sentence-transformer (cached in `results/emb_cache/`). Each pair is represented as `[u, v]`, the embeddings of the original and the perturbed text, plus a domain one-hot (CoT vs. Question), and a small MLP predicts the perturbation strength (0 = no/minimal change ... 3 = strong change). Because the input depends on the order of the two texts, `--swap-aug` (also train on the reversed pair) is worth trying. It reads all labelled pairs from `classifier_train_data/train.csv` and runs grouped, stratified k-fold CV (pairs with the same original text never straddle folds; a given seed always gives the same folds), reporting linear-weighted kappa (primary), Pearson correlation, accuracy, macro-F1 and MAE, overall and per domain.

```bash
pip install torch sentence-transformers scikit-learn pandas
python train_cv_mlp.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --hidden 32 --swap-aug --seeds 42 43 44 45 46 --final-fit --output-dir results/mlp_minilm
python train_cv_mlp.py --pca 32 --hidden 16 --seeds 42 43 44 45 46   # reduce [u, v] to 32 components first
```

### Evaluating on the test set
`evaluate_mlp.py` loads a final model from `train_cv_mlp.py --final-fit`, embeds the pairs in `annotation_data/test.csv` with the same embedding model, and reports linear-weighted kappa and Pearson correlation (plus accuracy, macro-F1, MAE) overall, per domain and per source file, with a confusion matrix. Test pairs whose original text also appears in the training data are flagged, and metrics are also reported without them.

It then predicts every text pair in every sheet of the Excel files in `annotation_data` (including rows without a human score) and writes the prediction to a column `mlp` (reused if it exists, otherwise added after the last column). Files are saved in place; use `--excel-out-dir DIR` to write copies instead, `--excel-column NAME` for another column name, or `--no-excel` to skip this step.

```bash
python evaluate_mlp.py --model-path results/mlp_minilm/final_model.joblib
```
