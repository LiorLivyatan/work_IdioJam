#!/usr/bin/env python3
"""
Re-evaluate hard_idioms metrics from filtered responses.json files.

Bypasses wandb/run_exp_hard.py entirely — just loads responses + test data,
calls process_responses, and writes metrics.json / results.tsv / conf_matrices_reports.txt
back to the same directory.

Usage:
    python scripts/reeval_hard_idioms.py          # re-eval all english hard_idioms runs
    python scripts/reeval_hard_idioms.py --dry_run  # print paths only, don't write
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# --- project root on sys.path ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from src.hard_idioms import get_data, process_responses
from src.utils import calc_metrics_classification


def reeval_run(run_dir: Path, dry_run: bool = False) -> None:
    config_path = run_dir / "config.yaml"
    responses_path = run_dir / "responses.json"

    if not config_path.exists() or not responses_path.exists():
        print(f"  SKIP {run_dir}: missing config.yaml or responses.json")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    sc_runs = config.get("sc_runs", 1)
    lang = config.get("lang", "english")

    with open(responses_path, encoding="utf-8-sig") as f:
        responses = json.load(f)

    # Load updated test data (now 534 rows after Phase 1 cleanup)
    test = get_data(lang=lang, data_path=config.get("data_path"))
    if "language" in test.columns:
        test = test[test["language"] == lang]

    if dry_run:
        print(f"  DRY  {run_dir.relative_to(ROOT)}: responses={len(responses)} test={len(test)}")
        return

    print(f"  EVAL {run_dir.relative_to(ROOT)}: responses={len(responses)} test={len(test)}", end="", flush=True)

    metrics, test_out, run_res, log_cm_report = process_responses(
        responses,
        test,
        calc_metrics_classification,
        lang=lang,
        sc_runs=sc_runs,
    )

    # Write metrics.json
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Write conf_matrices_reports.txt
    if log_cm_report:
        with open(run_dir / "conf_matrices_reports.txt", "w") as f:
            for lng, val in log_cm_report.items():
                f.write(f"Language: {lng}\n")
                f.write(f"Confusion matrix:\n{val['conf_matrix']}\n\n")
                f.write(f"Classification report:\n{val['report']}\n\n")

    # Write results.tsv
    test_out.to_csv(run_dir / "results.tsv", index=False, sep="\t")

    f1 = metrics.get("english", metrics).get("f1", metrics.get("f1", "?"))
    print(f"  -> F1={f1:.4f}" if isinstance(f1, float) else f"  -> F1={f1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base = ROOT / "results" / "hard_idioms" / "english" / "updated"
    run_dirs = sorted(p.parent for p in base.rglob("responses.json"))
    print(f"Found {len(run_dirs)} hard_idioms run directories")

    for run_dir in run_dirs:
        reeval_run(run_dir, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
