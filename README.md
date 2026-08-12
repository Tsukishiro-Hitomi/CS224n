# CS224n

Solutions to the CS224n (Natural Language Processing with Deep Learning, version: 2025 Winter)
assignments. Each assignment directory holds the starter code plus the LaTeX
source of the written answers.

## Assignment 1 — Exploring Word Vectors (25 pts)

A single Jupyter notebook, `a1/student/exploring_word_vectors.ipynb`.

- **Count-based word vectors** (10) — build a co-occurrence matrix, reduce it
  with truncated SVD, and plot the result.
- **Prediction-based word vectors** (15) — the same analysis on pretrained
  word2vec embeddings, then examine synonyms, antonyms, analogies, and bias.

## Assignment 2 — word2vec and Dependency Parsing

Written derivations in `a2_tex/`, code in `a2/a2/`.

- **Understanding word2vec** (20) — derive the gradients of the naive-softmax
  and negative-sampling losses for skip-gram.
- **Neural network optimization** (8) — momentum, Adam, and dropout.
- **Neural transition-based dependency parsing** (40) — implement the transition
  mechanics (`parser_transitions.py`) and the neural parser that picks
  transitions (`parser_model.py`), then train it with `run.py`.

## Assignment 3 — Transformers

Written answers in `a3_tex/assignment_3/`, code in `a3/cs224n-A3-student/`.

- **Attention exploration** (14) — properties of attention as an averaging
  operation, and what single-head attention cannot do.
- **Position embeddings exploration** (6) — why a bag-of-vectors self-attention
  layer is permutation-invariant, and what position embeddings add.
- **Coding a transformer from scratch** (30) — build the model in
  `model_solution.py` and train it with `train.py`; `pytest` tests included.

## Assignment 4 — LLM Evals

Written answers in `a4_tex/`, code in `a4/cs224n-A4-student/`. Unlike the earlier
assignments this one trains nothing: it queries a hosted model and evaluates it.

- **Standard benchmarking** (10) — benchmark two models on 100 GSM8K problems by
  string match, then design a prompt that beats the provided one (`gsm8k.py`).
- **LLM-as-a-judge** (13) — have a third model pick the better of two responses
  on 30 AlpacaEval prompts, compare its verdicts against my own, and test
  whether it is really measuring quality (`llm_judge.py`).
- **Red-teaming** (12) — make three models with progressively stronger system
  prompts reveal a password they were told to keep secret (`redteam.py`).

### Note on the A4 setup

The assignment expects Google Vertex AI via GCP credits issued to enrolled
students. This copy instead talks to any **OpenAI-compatible endpoint** serving
Gemini, configured in `a4/cs224n-A4-student/.env`:

```
LLM_BASE_URL=...      # relay endpoint, up to and including /v1
LLM_API_KEY=...
LLM_MODEL=google/gemini-2.5-flash-lite
STUDENT_EMAIL=...     # seeds the per-student passwords in the red-teaming task
```

Only `GeminiProvider.__init__` and `query` in `client/providers.py` were changed;
the model letters A–J and their hidden system prompts are untouched. Run the
scripts with the local virtualenv, e.g. `.venv/bin/python gsm8k.py`.

The evaluation outputs are committed (`llm_judge_results*.jsonl`) because
sampling is non-deterministic — re-running produces different numbers than the
ones cited in the write-up.
