# CS224n Assignment 4 - LLM Evals

In this assignment, you will evaluate the properties of various different LLMs using
standard benchmarking techniques, implement an LLM-as-a-judge evaluation, and explore
red-teaming approaches.

## Install

First, create a new conda environment with Python 3.10:

```bash
conda create -n cs224n-A4 python=3.10
```

Activate the environment:

```bash
conda activate cs224n-A4
```

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

**Set up your environment:**

This assignment uses **Gemini** through an OpenAI-compatible relay endpoint.

Create a local `.env` file (or export env vars in your shell) with:

- `LLM_BASE_URL`: base URL of the relay, up to and including `/v1`
- `LLM_API_KEY`: your relay API key
- `LLM_MODEL`: model name to query (defaults to `gemini-2.5-flash`)
- `STUDENT_EMAIL`: only needed for models **G/H/I** (used to deterministically seed the per-student password)

For an example see `example_usage.py`.

**Run the example:**

```bash
python example_usage.py
```
