# BFCL V4 comparison

_Evaluated 2026-05-13, n=458_

| Model | Overall accuracy |
| --- | --- |
| Base (Qwen/Qwen3.5-4B) | 87.3% |
| **Fine-tuned adapter** | **79.0%** |
| Delta | **-8.3pp** |

## Per-category breakdown

| Category | Base | Tuned | Delta |
| --- | --- | --- | --- |
| irrelevance | 80.0% | 42.0% | -38.0pp |
| live_irrelevance | 98.0% | 78.0% | -20.0pp |
| live_multiple | 78.0% | 78.0% | +0.0pp |
| live_parallel | 81.2% | 68.8% | -12.5pp |
| live_parallel_multiple | 95.8% | 91.7% | -4.2pp |
| live_relevance | 66.7% | 77.8% | +11.1pp |
| live_simple | 80.0% | 74.0% | -6.0pp |
| multiple | 92.0% | 90.0% | -2.0pp |
| parallel | 88.0% | 88.0% | +0.0pp |
| parallel_multiple | 98.0% | 92.0% | -6.0pp |
| simple | 90.0% | 88.0% | -2.0pp |
