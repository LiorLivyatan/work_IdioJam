#!/usr/bin/env python3
"""
Re-evaluate id10m metrics from filtered responses.json files.

Usage:
    python scripts/reeval_id10m.py
    python scripts/reeval_id10m.py --dry_run
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from src.id10m_utils import ID10M_UTILS
from src.utils import calc_metrics_classification

FINAL_JSON = ROOT / "data" / "raw_id10m_data" / "english" / "id10m_english_FINAL.json"


def load_test_df() -> pd.DataFrame:
    data = json.loads(FINAL_JSON.read_text(encoding="utf-8-sig"))
    return pd.DataFrame(data)


def reeval_run(run_dir: Path, test_df: pd.DataFrame, dry_run: bool = False) -> None:
    config_path = run_dir / "config.yaml"
    responses_path = run_dir / "responses.json"

    if not config_path.exists() or not responses_path.exists():
        print(f"  SKIP {run_dir}: missing files")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    sc_runs = config.get("sc_runs", 1)
    lang = config.get("lang", "english")

    with open(responses_path, encoding="utf-8-sig") as f:
        responses = json.load(f)

    if dry_run:
        print(f"  DRY  {run_dir.relative_to(ROOT)}: responses={len(responses)} test={len(test_df)}")
        return

    print(f"  EVAL {run_dir.relative_to(ROOT)}: responses={len(responses)} test={len(test_df)}", end="", flush=True)

    metrics, test_out, _, log_cm_report = ID10M_UTILS["process_responses"](
        responses,
        test_df.copy(),
        calc_metrics_classification,
        lang=lang,
        sc_runs=sc_runs,
    )

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    test_out.to_csv(run_dir / "results.tsv", index=False, sep="\t")

    if log_cm_report:
        with open(run_dir / "conf_matrices_reports.txt", "w") as f:
            for lng, val in log_cm_report.items():
                f.write(f"Language: {lng}\n")
                f.write(f"Confusion matrix:\n{val['conf_matrix']}\n\n")
                f.write(f"Classification report:\n{val['report']}\n\n")

    f1 = metrics.get("english", metrics).get("f1", metrics.get("f1", "?"))
    print(f"  -> F1={f1:.4f}" if isinstance(f1, float) else f"  -> F1={f1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    test_df = load_test_df()
    print(f"Test data: {len(test_df)} rows")

    base = ROOT / "results" / "id10m" / "english" / "updated"
    run_dirs = sorted(p.parent for p in base.rglob("responses.json"))
    print(f"Found {len(run_dirs)} id10m run directories")

    for run_dir in run_dirs:
        reeval_run(run_dir, test_df, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
