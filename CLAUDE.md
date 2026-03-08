# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a thesis research project (IdioJam) for LLM-based idiom detection. It has two main components:
1. **Data generation** (`data/`): Generate "confusing context" variants of idiom-containing sentences using LLMs
2. **Experiments** (`experiments/`): Run LLM idiom detection tasks and evaluate results

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Create `keys.yaml` in the repo root with API keys (this file is not committed):
```yaml
OPENAI_API_KEY: "..."
GOOGLE_API_KEY: "..."
ANTHROPIC_API_KEY: "..."
TOGETHER_API_KEY: "..."
```

## Running Experiments

All commands run from the repo root. The experiment runner reads `config.yaml` by default.

```bash
# Standard tasks (id10m, coam, parseme, lcp_multi, etc.)
python experiments/run_exp.py

# Hard idioms task (uses run-numbered subdirectories: run_001, run_002, ...)
python experiments/run_exp_hard.py

# Override config values via CLI
python experiments/run_exp_hard.py --seed 43 --lang german --sc_runs 5
python experiments/run_exp_hard.py --config_file my_config.yaml

# Re-evaluate existing responses without calling the API
python experiments/run_exp_hard.py --responses_dir experiments/logs/hard_idioms/german/hard_idioms_gpt-4o-mini_.../run_001
```

**Debug mode**: Set `debug: True` in `config.yaml` to run on 5 samples and skip W&B logging.

**Self-consistency**: Set `sc_runs: 5` and `temperature: 0.8` together. With `sc_runs: 1`, use `temperature: 0.3`.

## Data Generation

Run from `data/scripts/`:
```bash
python variants_generator.py
```
Uses Gemini 2.5 Pro via the `agno` framework. Requires `GEMINI_API_KEY` in `.env`.

## Architecture

### Task Utils Pattern

Each task is encapsulated in a utils module (`experiments/src/*_utils.py`) that exports a dict with exactly 4 keys:
```python
TASK_UTILS = {
    "get_data": get_data,           # Returns (train, test) DataFrames
    "get_prompt_schema": get_prompt_schema,  # Returns (ChatPromptTemplate, schema)
    "get_user_inputs": get_user_inputs,      # Returns list of dicts for chain.batch()
    "process_responses": process_responses,  # Parses LLM responses, computes metrics
}
```

The experiment runners (`run_exp.py`, `run_exp_hard.py`) select the right utils via `get_task_utils(task_name)`.

### Schema Selection

- **Pydantic schemas** (`pydantic_schemas.py`): Used for GPT, Gemini, and o-series models (support `.with_structured_output()`)
- **TypedDict schemas** (`typed_schemas.py`): Used for Anthropic (Claude) and Together.ai models

The `get_prompt_schema()` function in each utils module selects the appropriate schema set based on the model name.

### LLM Provider Routing (`experiments/src/models.py`)

`get_model()` routes based on model name substring:
- `"gemini"` → `ChatGoogleGenerativeAI`
- `"gpt"` → `ChatOpenAI`
- `"o1"`, `"o3"` → `ChatOpenAI` (no temperature)
- `"claude"` → `ChatAnthropic`
- anything else → `init_chat_model(..., model_provider="together")`

### Output Structure

```
experiments/
  results/{task}/{lang}/full_results.csv      # Aggregated results across all runs
  logs/{task}/{lang}/{exp_name}/              # Per-experiment directory (run_exp.py)
  logs/{task}/{lang}/{exp_name}/run_001/      # Per-run directory (run_exp_hard.py)
    config.yaml
    responses.json
    metrics.json
    results.tsv
    conf_matrices_reports.txt
```

### Supported Tasks

| Task name | Utils module | Metric type |
|-----------|-------------|-------------|
| `hard_idioms` | `hard_idioms.py` | classification (BIO tagging) |
| `id10m` | `id10m_utils.py` | classification (BIO tagging) |
| `coam` | `coam_utils.py` | MWE span matching |
| `parseme` | `parseme_utils.py` | MWE span matching |
| `parseme_vid` | `parseme_vid_utils.py` | MWE span matching |
| `lcp_multi`, `lcp_single` | `lcp_utils.py` | regression |
| `magpie_mini` | `magpie_utils.py` | classification |

### Key Config Parameters (`config.yaml`)

- `task`: task name (see table above)
- `lang`: `english` or `german` (task-dependent)
- `model`: model string (see `models.py` for routing)
- `prompt_type`: `zero_shot`, `few_shot_cot_best`, etc.
- `shots`: number of few-shot examples (must be even; 0 for zero-shot)
- `sc_runs`: self-consistency runs (1 = disabled, use 0.3 temp; >1 = SC, use 0.8 temp)
- `debug`: if True, runs on 5 samples, skips W&B
- `responses_dir`: if set, skips model inference and re-evaluates existing responses
- `data_path`: override default data file path (for correction runs)

### Munkres / Hungarian Algorithm

`experiments/src/bmc_munkres/munkres.py` is a vendored Hungarian algorithm implementation used in `calc_metrics_mwe()` for optimal token-level MWE matching.
