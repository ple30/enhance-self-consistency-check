# Human-annotated data

Samples are drawn from the [minimal_mistake_results](https://github.com/technion-cs-nlp/parametric-faithfulness/tree/main/minimal_mistake_results) of the parametric-faithfulness repository. The sampling code will be added later.

Files follow the pattern `<dataset>_<cot|input>_annotation.xlsx`.

| Column | Description |
|---|---|
| `score` | Perturbation strength from the primary annotator |
| `annotator 2 score` | Perturbation strength from the second annotator, used to check agreement with the primary annotator |
| `gemini` | Perturbation strength from the LLM judge approach, using Gemini |
| `cosine distance` | Cosine distance between the original and perturbed text |
| `surprisal` | Change in surprisal between the original and perturbed text |
| `mlp` | Output of a method that is no longer used; kept for reference |
