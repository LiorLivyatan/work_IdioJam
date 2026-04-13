"""
Complete the English hard_idioms few_shot claude-sonnet-4-20250514 SC run.

Current state: 561 items with 4/5 SC responses (batch 4 failed due to credit limit).

Strategy:
  - 456 UNCHANGED sentences: replace current 4 SC responses with the 5 from the archive run
  - 105 NEW/CHANGED sentences: run 1 more SC batch (batch 4) to get the 5th response
  - Result: 561 items with 5/5 SC responses

Usage:
    python experiments/complete_english_few_shot_sc.py
    python experiments/complete_english_few_shot_sc.py --skip_inference  # merge only, no API call
"""

import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from transformers import set_seed

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from src.utils import set_keys, parse_response, calc_metrics_classification
from src.models import get_model
from src.hard_idioms import HARD_IDIOMS_UTILS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
UPDATED_DIR  = REPO_ROOT / "results" / "hard_idioms" / "english" / "updated"
RUN_DIR      = UPDATED_DIR / "claude-sonnet-4-20250514" / "few_shot" / "seed_42"
ARCHIVE_DIR  = REPO_ROOT / "results" / "hard_idioms" / "english" / "archive" / "claude-sonnet-4-20250514" / "few_shot" / "seed_42"
FINAL_JSON   = REPO_ROOT / "data" / "hard_idioms_data" / "english" / "hard_idioms_english_FINAL.json"
SUBSET_JSON  = REPO_ROOT / "data" / "hard_idioms_data" / "english" / "hard_idioms_english_FINAL_subset105.json"
COMP_CSV     = REPO_ROOT / "data" / "hard_idioms_data" / "english" / "final_vs_old_comparison.csv"

MODEL        = "claude-sonnet-4-20250514"
LANG         = "english"
SC_RUNS      = 5
TEMPERATURE  = 0.8
PROMPT_TYPE  = "few_shot_cot_best"
SHOTS        = 10
SEED         = 42


def save_json(obj, path: Path):
    def clean(o):
        if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):  return [clean(i) for i in o]
        if isinstance(o, float) and pd.isna(o): return None
        return o
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(clean(obj), f, indent=1, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip API call — merge only (requires responses_105_batch4.json)")
    args = parser.parse_args()

    with open(REPO_ROOT / "keys.yaml") as f:
        set_keys(yaml.safe_load(f))

    # ── Load comparison table ──────────────────────────────────────────────────
    comp = pd.read_csv(COMP_CSV)
    unchanged_vs = set(comp[comp["status"] == "UNCHANGED"]["variant_sentence"].str.strip())
    new_vs       = set(comp[comp["status"] != "UNCHANGED"]["variant_sentence"].str.strip())
    logger.info(f"UNCHANGED: {len(unchanged_vs)}, NEW: {len(new_vs)}")

    # ── Load current 4-SC responses ────────────────────────────────────────────
    with open(RUN_DIR / "responses.json", encoding="utf-8-sig") as f:
        current = json.load(f)
    logger.info(f"Current responses: {len(current)} items, "
                f"{set(len(r['responses']) for r in current)} SC each")

    # ── Load archive 5-SC responses ────────────────────────────────────────────
    with open(ARCHIVE_DIR / "responses.json", encoding="utf-8-sig") as f:
        archive = json.load(f)
    arch_by_final   = {r["final_variant"].strip(): r["responses"] for r in archive}
    arch_by_variant = {r["variant_sentence"].strip(): r["responses"] for r in archive}
    logger.info(f"Archive responses: {len(archive)} items")

    # ── Step 1: Run batch 4 on the 105 new sentences ───────────────────────────
    batch4_path = RUN_DIR / "responses_105_batch4.json"

    if not args.skip_inference:
        logger.info("Step 1: Running SC batch 4 on 105 new sentences...")
        set_seed(SEED)

        config = dict(
            task="hard_idioms", lang=LANG, model=MODEL, seed=SEED,
            sc_runs=1, temperature=TEMPERATURE,
            prompt_type=PROMPT_TYPE, shots=SHOTS,
            use_rate_limiter=False, batched=True, num_samples=5,
            data_path=None, responses_dir=None,
        )

        subset_df = pd.read_json(SUBSET_JSON)
        subset_df["language"] = LANG

        llm = get_model(MODEL, TEMPERATURE, False)
        prompt, schema = HARD_IDIOMS_UTILS["get_prompt_schema"](config=config, train=None)
        if schema:
            llm = llm.with_structured_output(schema, include_raw=True)
            structured = True
        else:
            structured = False

        chain = prompt | llm
        user_inputs = HARD_IDIOMS_UTILS["get_user_inputs"](subset_df)

        new_responses = []
        for _, row in subset_df.iterrows():
            new_responses.append({key: row[key] for key in subset_df.columns} | {"responses": []})

        try:
            raw = chain.batch(user_inputs, config={"max_concurrency": 1})
            logger.info(f"  Got {len(raw)} responses")
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            raise

        for i, resp in enumerate(raw):
            try:
                new_responses[i]["responses"].append(parse_response(resp, structured))
            except Exception as e:
                logger.error(f"  parse_response error row {i}: {e}")
                new_responses[i]["responses"].append({})

        save_json(new_responses, batch4_path)
        logger.info(f"Saved responses_105_batch4.json ({len(new_responses)} rows)")
    else:
        if not batch4_path.exists():
            raise FileNotFoundError(f"--skip_inference set but {batch4_path} not found")
        with open(batch4_path, encoding="utf-8-sig") as f:
            new_responses = json.load(f)
        logger.info(f"Loaded existing responses_105_batch4.json ({len(new_responses)} rows)")

    new_by_variant = {r["variant_sentence"].strip(): r["responses"][0] for r in new_responses}

    # ── Step 2: Build merged 5-SC responses ────────────────────────────────────
    logger.info("Step 2: Building merged 5-SC responses...")
    merged = []
    missing = 0

    for item in current:
        vs = item["variant_sentence"].strip()

        if vs in unchanged_vs:
            # Use all 5 responses from archive
            arch_resps = arch_by_final.get(vs) or arch_by_variant.get(vs)
            if arch_resps is None:
                logger.warning(f"UNCHANGED item missing from archive: {vs[:60]}")
                missing += 1
                merged.append({**item, "responses": item["responses"]})
            else:
                merged.append({**item, "responses": arch_resps})

        else:
            # New sentence: current 4 + 1 new from batch 4
            new_resp = new_by_variant.get(vs)
            if new_resp is None:
                logger.warning(f"NEW item missing from batch4: {vs[:60]}")
                missing += 1
                merged.append({**item})
            else:
                merged.append({**item, "responses": item["responses"] + [new_resp]})

    sc_counts = set(len(r["responses"]) for r in merged)
    logger.info(f"Merged: {len(merged)} items, SC counts: {sc_counts}, missing: {missing}")

    if sc_counts != {5}:
        logger.error(f"Not all items have 5 SC responses — aborting save")
        return

    save_json(merged, RUN_DIR / "responses.json")
    logger.info("Saved responses.json (5 SC, 561 items)")

    # ── Step 3: Re-evaluate ────────────────────────────────────────────────────
    logger.info("Step 3: Re-evaluating metrics...")
    config = dict(
        task="hard_idioms", lang=LANG, model=MODEL, seed=SEED,
        sc_runs=SC_RUNS, temperature=TEMPERATURE,
        prompt_type=PROMPT_TYPE, shots=SHOTS,
        use_rate_limiter=False, batched=True, num_samples=5,
        data_path=None, responses_dir=None,
    )

    test_df = pd.read_json(FINAL_JSON)
    test_df["language"] = LANG

    metrics, test_out, _, log_cm = HARD_IDIOMS_UTILS["process_responses"](
        merged, test_df, calc_metrics_classification,
        lang=LANG, sc_runs=SC_RUNS,
    )
    logger.info(f"Metrics: {metrics}")

    with open(RUN_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    test_out.to_csv(RUN_DIR / "results.tsv", index=False, sep="\t")

    if log_cm:
        with open(RUN_DIR / "conf_matrices_reports.txt", "w") as f:
            for lang_key, data in log_cm.items():
                f.write(f"Language: {lang_key}\n")
                f.write(f"Confusion matrix:\n{data['conf_matrix']}\n\n")
                f.write(f"Classification report:\n{data['report']}\n\n")

    # Update config.yaml to reflect final sc_runs=5
    cfg_path = RUN_DIR / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["sc_runs"] = SC_RUNS
    cfg["experiment_start_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    logger.info(f"Done. F1={metrics.get(LANG, {}).get('f1', '?'):.4f}")


if __name__ == "__main__":
    main()
