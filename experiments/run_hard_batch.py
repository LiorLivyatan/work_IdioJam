"""
Batch runner for updated hard_idioms English experiments.

For each experiment in the matrix:
  1. Run inference on the 105 new/changed rows  →  responses_105.json
  2. Merge with the 456 unchanged responses from the matching old run  →  responses.json
  3. Evaluate on the full 561-row FINAL dataset  →  metrics.json, results.tsv, etc.

All three outputs land in a single new run directory:
  experiments/logs/hard_idioms/english/{exp_name}/run_001/
      responses_105.json
      responses.json
      config.yaml
      metrics.json
      results.tsv
      conf_matrices_reports.txt

Usage:
    python experiments/run_hard_batch.py                        # run all 30 experiments
    python experiments/run_hard_batch.py --dry_run              # print plan, no API calls
    python experiments/run_hard_batch.py --filter gpt-4o-mini   # only configs matching substring
    python experiments/run_hard_batch.py --skip_inference       # merge+eval only
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from transformers import set_seed

# Add repo root to path so src.* imports work
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from src.utils import set_keys, parse_response, calc_metrics_classification
from src.models import get_model
from src.hard_idioms import HARD_IDIOMS_UTILS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

####################################################################################################
# Paths

RESULTS_DIR    = REPO_ROOT / "results" / "hard_idioms" / "english"
ARCHIVE_DIR    = RESULTS_DIR / "archive"    # old runs (pre-data-fix)
UPDATED_DIR    = RESULTS_DIR / "updated"    # new runs (post-data-fix, this batch)
FINAL_JSON     = REPO_ROOT / "data" / "hard_idioms_data" / "english" / "hard_idioms_english_FINAL.json"
SUBSET_JSON    = REPO_ROOT / "data" / "hard_idioms_data" / "english" / "hard_idioms_english_FINAL_subset105.json"
COMPARISON_CSV = REPO_ROOT / "data" / "hard_idioms_data" / "english" / "final_vs_old_comparison.csv"

####################################################################################################
# Experiment matrix

PROMPT_CONFIGS = {
    "zero_shot": dict(prompt_type="zero_shot",        shots=0,  sc_runs=1, temperature=0.3),
    "few_shot":  dict(prompt_type="few_shot_cot_best", shots=10, sc_runs=5, temperature=0.8),
}

EXPERIMENT_MATRIX = [
    # Standard models — zero_shot + few_shot, seeds 42/43/44
    dict(model="gemini-2.5-flash-lite",                     prompt_types=["zero_shot", "few_shot"], seeds=[42, 43, 44]),
    dict(model="gpt-4o",                                    prompt_types=["zero_shot", "few_shot"], seeds=[42]),
    dict(model="gpt-4o-mini",                               prompt_types=["zero_shot", "few_shot"], seeds=[42, 43, 44]),
    dict(model="meta-llama/Llama-4-Scout-17B-16E-Instruct", prompt_types=["zero_shot", "few_shot"], seeds=[42, 43, 44]),
    dict(model="Qwen/Qwen2.5-72B-Instruct-Turbo",           prompt_types=["zero_shot", "few_shot"], seeds=[42, 43, 44]),
    # Premium models — zero_shot + few_shot, seed 42 only
    dict(model="gemini-2.5-pro",                            prompt_types=["zero_shot", "few_shot"], seeds=[42]),
    dict(model="o3-mini",                                   prompt_types=["zero_shot"],             seeds=[42]),
    dict(model="deepseek-ai/DeepSeek-R1",                   prompt_types=["zero_shot"],             seeds=[42]),
]

####################################################################################################
# Helpers


def exp_name(model: str, prompt_cfg: dict, seed: int, lang: str = "english") -> str:
    model_name = model.split("/")[-1]
    return (
        f"hard_idioms_{model_name}_{prompt_cfg['prompt_type']}"
        f"_shots_{prompt_cfg['shots']}"
        f"_sc{prompt_cfg['sc_runs']}"
        f"_tmp{prompt_cfg['temperature']}"
        f"_seed{seed}_{lang}"
    )


def _model_dir_name(model: str) -> str:
    return model.split("/")[-1]


def _prompt_dir_name(prompt_cfg: dict) -> str:
    return "zero_shot" if prompt_cfg["prompt_type"] == "zero_shot" else "few_shot"


def find_old_run_dir(model: str, prompt_cfg: dict, seed: int) -> Path | None:
    """Return the archive directory for this experiment, or None if not found."""
    candidate = ARCHIVE_DIR / _model_dir_name(model) / _prompt_dir_name(prompt_cfg) / f"seed_{seed}"
    return candidate if candidate.exists() else None


def create_run_dir(model: str, prompt_cfg: dict, seed: int) -> Path:
    """Create (or return existing) updated/{model}/{prompt}/seed_{N}/ directory."""
    run_dir = UPDATED_DIR / _model_dir_name(model) / _prompt_dir_name(prompt_cfg) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_config(model: str, prompt_cfg: dict, seed: int) -> dict:
    return dict(
        task="hard_idioms",
        lang="english",
        model=model,
        seed=seed,
        debug=False,
        sc_runs=prompt_cfg["sc_runs"],
        temperature=prompt_cfg["temperature"],
        prompt_type=prompt_cfg["prompt_type"],
        shots=prompt_cfg["shots"],
        use_rate_limiter=False,
        batched=True,
        results_dir=str(REPO_ROOT / "experiments" / "results"),
        logs_dir=str(REPO_ROOT / "experiments" / "logs"),
        data_path=None,
        responses_dir=None,
        num_samples=5,
    )


def run_inference_on_subset(config: dict) -> list[dict]:
    """
    Run the model on the 105-row subset. Returns a list of response dicts
    (same structure as responses.json: each entry has metadata + 'responses' key).
    """
    set_seed(config["seed"])

    test = HARD_IDIOMS_UTILS["get_data"](
        task=config["task"],
        lang=config["lang"],
        data_path=str(SUBSET_JSON),
    )
    test["language"] = config["lang"]

    llm = get_model(config["model"], config["temperature"], config["use_rate_limiter"])
    prompt, schema = HARD_IDIOMS_UTILS["get_prompt_schema"](config=config, train=None)

    if schema:
        llm = llm.with_structured_output(schema, include_raw=True)
        structured = True
    else:
        structured = False

    chain = prompt | llm
    user_inputs = HARD_IDIOMS_UTILS["get_user_inputs"](test)

    # Build response containers
    responses = []
    for _, row in test.iterrows():
        responses.append({key: row[key] for key in test.columns} | {"responses": []})

    for run_index in range(config["sc_runs"]):
        logger.info(f"  SC run {run_index + 1}/{config['sc_runs']}")
        if config["batched"]:
            try:
                raw = chain.batch(user_inputs)
            except Exception as e:
                logger.error(f"Batch run {run_index} failed: {e}")
                continue
        else:
            raw = []
            for i, inp in enumerate(user_inputs):
                try:
                    raw.append(chain.invoke(inp))
                except Exception as e:
                    logger.warning(f"  Row {i} failed: {e}")
                    raw.append(None)

        for i, resp in enumerate(raw):
            try:
                responses[i]["responses"].append(parse_response(resp, structured))
            except Exception as e:
                logger.error(f"  parse_response error row {i}: {e}")
                responses[i]["responses"].append({})

    return responses


def build_merged_responses(old_run_dir: Path, new_responses: list[dict]) -> list[dict]:
    """
    Merge old (456 unchanged) + new (105 changed) into FINAL order (561 rows).
    Keys by variant_sentence text. Old responses use the 'final_variant' field.
    """
    with open(FINAL_JSON) as f:
        final_rows = json.load(f)

    with open(old_run_dir / "responses.json", encoding="utf-8-sig") as f:
        old_list = json.load(f)
    old_by_variant = {r["final_variant"].strip(): r for r in old_list}

    new_by_variant = {r["variant_sentence"].strip(): r for r in new_responses}

    comp = pd.read_csv(COMPARISON_CSV)
    unchanged = set(comp[comp["status"] == "UNCHANGED"]["variant_sentence"].str.strip())

    merged, missing_old, missing_new = [], 0, 0
    for row in final_rows:
        vs = row["variant_sentence"].strip()
        if vs in unchanged:
            old_rec = old_by_variant.get(vs)
            if old_rec is None:
                logger.warning(f"UNCHANGED row missing from old: {vs[:60]}")
                missing_old += 1
                merged.append({**row, "language": "english", "responses": []})
            else:
                merged.append({**row, "language": "english", "responses": old_rec["responses"]})
        else:
            new_rec = new_by_variant.get(vs)
            if new_rec is None:
                logger.warning(f"NEW row missing from new: {vs[:60]}")
                missing_new += 1
                merged.append({**row, "language": "english", "responses": []})
            else:
                merged.append({**row, "language": "english", "responses": new_rec["responses"]})

    logger.info(f"Merged {len(merged)} rows (missing_old={missing_old}, missing_new={missing_new})")
    return merged


def evaluate_and_save(merged: list[dict], config: dict, run_dir: Path):
    """Run process_responses on the full 561-row merged data and save all outputs."""
    # Load full FINAL test data (561 rows) — provides ground-truth tokens/tags
    test = HARD_IDIOMS_UTILS["get_data"](
        task=config["task"],
        lang=config["lang"],
        data_path=str(FINAL_JSON),
    )
    test["language"] = config["lang"]

    metrics, test_out, run_res, log_cm_report = HARD_IDIOMS_UTILS["process_responses"](
        merged, test, calc_metrics_classification,
        lang=config["lang"],
        sc_runs=config["sc_runs"],
    )
    logger.info(f"Metrics: {metrics}")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    test_out.to_csv(run_dir / "results.tsv", index=False, sep="\t")

    if log_cm_report:
        with open(run_dir / "conf_matrices_reports.txt", "w") as f:
            for lang, data in log_cm_report.items():
                f.write(f"Language: {lang}\n")
                f.write(f"Confusion matrix:\n{data['conf_matrix']}\n\n")
                f.write(f"Classification report:\n{data['report']}\n\n")

    return metrics


def save_json(obj, path: Path):
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list): return [clean(i) for i in o]
        if isinstance(o, float) and pd.isna(o): return None
        return o
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(clean(obj), f, indent=1, ensure_ascii=False)


####################################################################################################
# Main


def expand_matrix() -> list[dict]:
    runs = []
    for entry in EXPERIMENT_MATRIX:
        for pt in entry["prompt_types"]:
            for seed in entry["seeds"]:
                runs.append(dict(model=entry["model"], prompt_cfg=PROMPT_CONFIGS[pt], seed=seed))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run",        action="store_true", help="Print plan only, no API calls")
    parser.add_argument("--filter",         type=str, default=None, help="Only run experiments whose name contains this string")
    parser.add_argument("--skip_inference", action="store_true", help="Merge+eval only — assumes responses_105.json already exists")
    args = parser.parse_args()

    # Load API keys
    with open(REPO_ROOT / "keys.yaml") as f:
        keys = yaml.safe_load(f)
    set_keys(keys)

    runs = expand_matrix()
    logger.info(f"Total experiments in matrix: {len(runs)}")

    for run in runs:
        model, prompt_cfg, seed = run["model"], run["prompt_cfg"], run["seed"]
        name = exp_name(model, prompt_cfg, seed)

        if args.filter and args.filter not in name:
            continue

        logger.info(f"\n{'='*64}\n{name}\n{'='*64}")

        old_run_dir = find_old_run_dir(model, prompt_cfg, seed)
        if old_run_dir is None:
            logger.warning(f"No archive run found — evaluating on 105 rows only")

        config = build_config(model, prompt_cfg, seed)

        if args.dry_run:
            dest = UPDATED_DIR / _model_dir_name(model) / _prompt_dir_name(prompt_cfg) / f"seed_{seed}"
            logger.info(f"  [DRY RUN] → {dest}")
            logger.info(f"  [DRY RUN]   archive: {old_run_dir}")
            continue

        # --- Create run directory ---
        run_dir = create_run_dir(model, prompt_cfg, seed)
        config["experiment_start_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        logger.info(f"Run dir: {run_dir}")

        # --- Step 1: Inference on 105-row subset ---
        if not args.skip_inference:
            logger.info("Step 1: Running inference on 105-row subset...")
            new_responses = run_inference_on_subset(config)
            save_json(new_responses, run_dir / "responses_105.json")
            logger.info(f"Saved responses_105.json ({len(new_responses)} rows)")
        else:
            path_105 = run_dir / "responses_105.json"
            if not path_105.exists():
                logger.error(f"--skip_inference set but {path_105} not found — skipping")
                continue
            with open(path_105, encoding="utf-8-sig") as f:
                new_responses = json.load(f)
            logger.info(f"Loaded existing responses_105.json ({len(new_responses)} rows)")

        # --- Step 2: Merge old 456 + new 105 ---
        if old_run_dir is not None:
            logger.info("Step 2: Merging old (456) + new (105) responses...")
            merged = build_merged_responses(old_run_dir, new_responses)
        else:
            logger.info("Step 2: No old run — evaluating on 105 rows only")
            merged = new_responses

        if old_run_dir is not None and len(merged) != 561:
            logger.error(f"Expected 561 merged rows, got {len(merged)} — aborting this experiment")
            continue

        save_json(merged, run_dir / "responses.json")
        logger.info(f"Saved responses.json ({len(merged)} rows)")

        # --- Step 3: Evaluate on merged 561 ---
        logger.info("Step 3: Evaluating...")
        try:
            metrics = evaluate_and_save(merged, config, run_dir)
            logger.info(f"Done → {run_dir}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback; traceback.print_exc()

    logger.info("\nAll experiments complete.")


if __name__ == "__main__":
    main()
