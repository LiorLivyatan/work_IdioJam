"""
Batch runner for updated id10m English experiments.

For each experiment in the archive:
  1. Run inference on the 8 corrected sentences (typos fixed in current dataset)
  2. Merge with the 179 unchanged responses from the archive run
  3. Evaluate on the full 187-sentence FINAL dataset

All outputs land in:
  results/id10m/english/updated/{model}/{prompt}/seed_{N}/
      responses_8.json      # new responses for the 8 corrected sentences
      responses.json        # merged 187-row dataset
      config.yaml
      metrics.json
      results.tsv
      conf_matrices_reports.txt

Usage:
    python experiments/run_id10m_batch_english.py               # run all 38 experiments
    python experiments/run_id10m_batch_english.py --dry_run      # print plan only
    python experiments/run_id10m_batch_english.py --filter gpt-4o-mini
    python experiments/run_id10m_batch_english.py --skip_inference  # merge+eval only
"""

import os
import sys
import csv
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from transformers import set_seed

# Add experiments/ dir to path so src.* imports work
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from src.utils import set_keys, parse_response, calc_metrics_classification
from src.models import get_model
from src.id10m_utils import ID10M_UTILS, read_bio_tsv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

####################################################################################################
# Paths

RESULTS_DIR     = REPO_ROOT / "results" / "id10m" / "english"
ARCHIVE_DIR     = RESULTS_DIR / "archive"
UPDATED_DIR     = RESULTS_DIR / "updated"
UPDATED_SUMMARY = UPDATED_DIR / "results_summary.csv"

# Pre-built JSON data files (see data/raw_id10m_data/english/)
FINAL_JSON   = REPO_ROOT / "data" / "raw_id10m_data" / "english" / "id10m_english_FINAL.json"
SUBSET_JSON  = REPO_ROOT / "data" / "raw_id10m_data" / "english" / "id10m_english_subset8.json"
TRAIN_TSV    = REPO_ROOT / "data" / "id10m" / "trainset" / "english.tsv"

LANG = "english"

####################################################################################################
# Experiment matrix — mirrors exactly the 38 archive runs

PROMPT_CONFIGS = {
    "zero_shot": dict(prompt_type="zero_shot",         shots=0,  sc_runs=1, temperature=0.3),
    "few_shot":  dict(prompt_type="few_shot_cot_best", shots=10, sc_runs=5, temperature=0.8),
}

EXPERIMENT_MATRIX = [
    dict(model="claude-3-haiku-20240307",      prompt_types=["zero_shot", "few_shot"], seeds=[42, 43, 44]),
    dict(model="claude-sonnet-4-20250514",     prompt_types=["zero_shot", "few_shot"], seeds=[42]),
    dict(model="deepseek/deepseek-r1",         prompt_types=["zero_shot"],             seeds=[42]),
    dict(model="gemini-2.5-flash-lite",        prompt_types=["zero_shot", "few_shot"], seeds=[42, 43, 44]),
    dict(model="gemini-2.5-pro",               prompt_types=["zero_shot", "few_shot"], seeds=[42]),
    dict(model="gpt-4o",                       prompt_types=["zero_shot", "few_shot"], seeds=[42]),
    dict(model="gpt-4o-mini",                  prompt_types=["zero_shot", "few_shot"], seeds=[42, 43, 44]),
    dict(model="meta-llama/llama-4-scout",     prompt_types=["zero_shot", "few_shot"], seeds=[42, 43, 44]),
    dict(model="o3-mini",                      prompt_types=["zero_shot"],             seeds=[42]),
    dict(model="qwen/qwen-2.5-72b-instruct",   prompt_types=["zero_shot", "few_shot"], seeds=[42, 43, 44]),
]

####################################################################################################
# Helpers


def _model_dir_name(model: str) -> str:
    return model.split("/")[-1]


def _prompt_dir_name(prompt_cfg: dict) -> str:
    return "zero_shot" if prompt_cfg["prompt_type"] == "zero_shot" else "few_shot"


def find_archive_dir(model: str, prompt_cfg: dict, seed: int) -> Path | None:
    dir_name = _model_dir_name(model)
    candidate = ARCHIVE_DIR / dir_name / _prompt_dir_name(prompt_cfg) / f"seed_{seed}"
    return candidate if candidate.exists() else None


def create_updated_dir(model: str, prompt_cfg: dict, seed: int) -> Path:
    run_dir = UPDATED_DIR / _model_dir_name(model) / _prompt_dir_name(prompt_cfg) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_config(model: str, prompt_cfg: dict, seed: int) -> dict:
    return dict(
        task="id10m",
        lang=LANG,
        model=model,
        seed=seed,
        debug=False,
        sc_runs=prompt_cfg["sc_runs"],
        temperature=prompt_cfg["temperature"],
        prompt_type=prompt_cfg["prompt_type"],
        shots=prompt_cfg["shots"],
        use_rate_limiter=False,
        batched=True,
        num_samples=5,
    )


####################################################################################################
# Data loading


def load_json_data(path: Path) -> pd.DataFrame:
    """Load a pre-built JSON data file into a DataFrame."""
    with open(path) as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def load_train_data() -> pd.DataFrame:
    """Load English train set from BIO TSV for few-shot examples."""
    df = read_bio_tsv(str(TRAIN_TSV))
    df["language"] = LANG
    return df


####################################################################################################
# Response normalization


def normalize_response(resp: dict) -> dict:
    """Convert old archive format {'idioms': [...]} to new format {'parsed': {'idioms': [...]}}."""
    if not resp:
        return resp
    if "parsed" in resp:
        return resp
    if "idioms" in resp:
        return {"parsed": {"idioms": resp["idioms"]}}
    return resp


####################################################################################################
# Inference


def run_inference_on_subset(
    subset_df: pd.DataFrame,
    config: dict,
    train_df: pd.DataFrame,
) -> list[dict]:
    """Run the model on the 8-sentence subset. Returns response dicts."""
    set_seed(config["seed"])

    llm = get_model(config["model"], config["temperature"], config["use_rate_limiter"])
    prompt, schema = ID10M_UTILS["get_prompt_schema"](config=config, train=train_df)

    if schema:
        llm = llm.with_structured_output(schema, include_raw=True)
        structured = True
    else:
        structured = False

    chain = prompt | llm
    user_inputs = ID10M_UTILS["get_user_inputs"](subset_df)

    responses = []
    for _, row in subset_df.iterrows():
        responses.append({key: row[key] for key in subset_df.columns} | {"responses": []})

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


####################################################################################################
# Merge


def build_merged_responses(
    archive_dir: Path,
    new_responses: list[dict],
    test_df: pd.DataFrame,
) -> list[dict]:
    """
    Build 187-row merged list in test_df order:
    - 179 rows: reused from archive (exact sentence match, old format normalized)
    - 8 rows: from new inference
    """
    with open(archive_dir / "responses.json", encoding="utf-8-sig") as f:
        archive_list = json.load(f)

    archive_by_sentence = {}
    for item in archive_list:
        key = item["sentence"].strip()
        normalized = [normalize_response(r) for r in item["responses"]]
        archive_by_sentence[key] = {**item, "responses": normalized}

    new_by_sentence = {r["sentence"].strip(): r for r in new_responses}

    missing_count = 0
    merged = []
    for _, row in test_df.iterrows():
        sentence = row["sentence"].strip()
        base = {col: row[col] for col in test_df.columns}

        if sentence in new_by_sentence:
            new_rec = new_by_sentence[sentence]
            merged.append({**base, "language": LANG, "responses": new_rec["responses"]})
        elif sentence in archive_by_sentence:
            arc = archive_by_sentence[sentence]
            merged.append({**base, "language": LANG, "responses": arc["responses"]})
        else:
            logger.warning(f"Sentence missing from both archive and new: {sentence[:70]}")
            missing_count += 1
            merged.append({**base, "language": LANG, "responses": []})

    logger.info(f"Merged {len(merged)} rows (missing={missing_count})")
    return merged


####################################################################################################
# Evaluate


def evaluate_and_save(
    merged: list[dict],
    config: dict,
    test_df: pd.DataFrame,
    run_dir: Path,
) -> dict:
    """Run process_responses on merged 187 rows and save all outputs."""
    metrics, test_out, _, log_cm_report = ID10M_UTILS["process_responses"](
        merged, test_df.copy(), calc_metrics_classification,
        lang=config["lang"],
        sc_runs=config["sc_runs"],
    )
    logger.info(f"Metrics: {metrics}")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    test_out.to_csv(run_dir / "results.tsv", index=False, sep="\t")

    if log_cm_report:
        with open(run_dir / "conf_matrices_reports.txt", "w") as f:
            for lang_key, data in log_cm_report.items():
                f.write(f"Language: {lang_key}\n")
                f.write(f"Confusion matrix:\n{data['conf_matrix']}\n\n")
                f.write(f"Classification report:\n{data['report']}\n\n")

    return metrics


####################################################################################################
# Summary CSV


def update_summary_csv(model: str, prompt_cfg: dict, seed: int, metrics: dict):
    SUMMARY_COLS = [
        "model", "prompt_type", "seed", "shots", "sc_runs", "temperature",
        "accuracy", "precision", "recall", "f1", "hallucinations",
    ]
    rows = []
    if UPDATED_SUMMARY.exists():
        with open(UPDATED_SUMMARY, newline="") as f:
            rows = list(csv.DictReader(f))

    prompt_type = _prompt_dir_name(prompt_cfg)
    eng = metrics.get(LANG, {})
    new_row = {
        "model":          model,
        "prompt_type":    prompt_type,
        "seed":           seed,
        "shots":          prompt_cfg["shots"],
        "sc_runs":        prompt_cfg["sc_runs"],
        "temperature":    prompt_cfg["temperature"],
        "accuracy":       round(eng.get("accuracy", 0.0), 6) if isinstance(eng.get("accuracy"), float) else "",
        "precision":      round(eng.get("precision", 0.0), 6) if isinstance(eng.get("precision"), float) else "",
        "recall":         round(eng.get("recall", 0.0), 6) if isinstance(eng.get("recall"), float) else "",
        "f1":             round(eng.get("f1", 0.0), 6) if isinstance(eng.get("f1"), float) else "",
        "hallucinations": eng.get("hallucinations", ""),
    }

    match_key = (model, prompt_type, str(seed))
    updated = False
    for i, r in enumerate(rows):
        if (r["model"], r["prompt_type"], str(r["seed"])) == match_key:
            rows[i] = new_row
            updated = True
            break
    if not updated:
        rows.append(new_row)

    with open(UPDATED_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Updated results_summary.csv ({len(rows)} rows)")


####################################################################################################
# JSON save helper


def save_json(obj, path: Path):
    def clean(o):
        if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):  return [clean(i) for i in o]
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
    parser.add_argument("--dry_run",        action="store_true", help="Print plan, no API calls")
    parser.add_argument("--filter",         type=str, default=None, help="Only configs matching substring")
    parser.add_argument("--skip_inference", action="store_true", help="Merge+eval only; requires responses_8.json")
    args = parser.parse_args()

    with open(REPO_ROOT / "keys.yaml") as f:
        keys = yaml.safe_load(f)
    set_keys(keys)

    logger.info("Loading test data (187 sentences)...")
    test_df = load_json_data(FINAL_JSON)
    logger.info(f"Test data: {len(test_df)} rows")

    logger.info("Loading subset data (8 corrected sentences)...")
    subset_df = load_json_data(SUBSET_JSON)
    logger.info(f"Subset: {len(subset_df)} sentences")
    for s in subset_df["sentence"].tolist():
        logger.info(f"  - {s}")

    logger.info("Loading train data (for few-shot)...")
    train_df = load_train_data()
    logger.info(f"Train data: {len(train_df)} rows")

    runs = expand_matrix()
    logger.info(f"Total experiments: {len(runs)}")

    for run in runs:
        model, prompt_cfg, seed = run["model"], run["prompt_cfg"], run["seed"]
        model_short = _model_dir_name(model)
        prompt_short = _prompt_dir_name(prompt_cfg)
        name = f"id10m_{model_short}_{prompt_cfg['prompt_type']}_shots_{prompt_cfg['shots']}_sc{prompt_cfg['sc_runs']}_seed{seed}_{LANG}"

        if args.filter and args.filter not in name:
            continue

        updated_run_dir = UPDATED_DIR / model_short / prompt_short / f"seed_{seed}"
        if (updated_run_dir / "metrics.json").exists():
            logger.info(f"Skipping {name} — already completed")
            continue

        logger.info(f"\n{'='*64}\n{name}\n{'='*64}")

        archive_dir = find_archive_dir(model, prompt_cfg, seed)
        if archive_dir is None:
            logger.warning(f"No archive run found for {model} {prompt_short} seed_{seed} — skipping")
            continue

        config = build_config(model, prompt_cfg, seed)

        if args.dry_run:
            logger.info(f"  [DRY RUN] archive : {archive_dir}")
            logger.info(f"  [DRY RUN] output  : {updated_run_dir}")
            logger.info(f"  [DRY RUN] subset  : {len(subset_df)} sentences to infer")
            continue

        # Create output directory and save config
        run_dir = create_updated_dir(model, prompt_cfg, seed)
        config_save = {**config, "experiment_start_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config_save, f)

        # --- Step 1: Inference on 8 corrected sentences ---
        if not args.skip_inference:
            logger.info(f"Step 1: Inference on {len(subset_df)} sentences...")
            new_responses = run_inference_on_subset(subset_df, config, train_df)
            save_json(new_responses, run_dir / "responses_8.json")
            logger.info(f"Saved responses_8.json ({len(new_responses)} rows)")
        else:
            path_8 = run_dir / "responses_8.json"
            if not path_8.exists():
                logger.error(f"--skip_inference set but {path_8} not found — skipping")
                continue
            with open(path_8, encoding="utf-8-sig") as f:
                new_responses = json.load(f)
            logger.info(f"Loaded existing responses_8.json ({len(new_responses)} rows)")

        # --- Step 2: Merge 179 archive + 8 new = 187 ---
        logger.info("Step 2: Merging archive (179) + new (8)...")
        merged = build_merged_responses(archive_dir, new_responses, test_df)

        if len(merged) != 187:
            logger.error(f"Expected 187 merged rows, got {len(merged)} — aborting this run")
            continue

        save_json(merged, run_dir / "responses.json")
        logger.info(f"Saved responses.json ({len(merged)} rows)")

        # --- Step 3: Evaluate on full 187-sentence set ---
        logger.info("Step 3: Evaluating on 187 sentences...")
        try:
            metrics = evaluate_and_save(merged, config, test_df.copy(), run_dir)
            update_summary_csv(model, prompt_cfg, seed, metrics)
            logger.info(f"Done → {run_dir}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback; traceback.print_exc()

    logger.info("\nAll experiments complete.")


if __name__ == "__main__":
    main()
