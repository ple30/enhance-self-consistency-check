# Human-annotated data

Samples are drawn from the [minimal_mistake_results](https://github.com/technion-cs-nlp/parametric-faithfulness/tree/main/minimal_mistake_results) of the parametric-faithfulness repository. The sampling code will be added later.

## Getting the source data

1. From the root of this repository, clone only the [minimal_mistake_results](https://github.com/technion-cs-nlp/parametric-faithfulness/tree/main/minimal_mistake_results) folder of the parametric-faithfulness repository, using `git clone` with a sparse checkout:

   ```bash
   git clone --filter=blob:none --sparse https://github.com/technion-cs-nlp/parametric-faithfulness.git
   cd parametric-faithfulness
   git sparse-checkout set minimal_mistake_results
   cd ..
   ```

   The folder is then at `parametric-faithfulness/minimal_mistake_results/`. The sparse checkout skips the repository's other folders. It still includes the small files at the top level of the repository, such as its README and LICENSE.

Files follow the pattern `<dataset>_<cot|input>_annotation.xlsx`.

| Column | Description |
|---|---|
| `score` | Perturbation strength from the primary annotator |
| `annotator 2 score` | Perturbation strength from the second annotator, used to check agreement with the primary annotator |
| `gemini` | Perturbation strength from the LLM judge approach, using [prompt_perturbation_strength.py](../prompt_perturbation_strength.py) |
| `cosine distance` | Cosine distance between the original and perturbed text |
| `surprisal` | Change in surprisal between the original and perturbed text |
| `mlp` | Output of a method that is no longer used; kept for reference |
