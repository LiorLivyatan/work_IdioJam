#!/usr/bin/env python3
"""
Generate a legacy-format comparison table with the same columns as the old
IdioGem/comparisons/{lang}/model_comparison_table.csv.

Reads from:
  - results/comparisons/{lang}/model_comparison_table.csv  (base metrics)
  - results/comparisons/{lang}/{model}/{prompt_type}/seed_{seed}/metrics.json
    (detailed per-run data: literal/idiomatic breakdown, per-idiom rates, examples)

Outputs:
  - results/comparisons/{lang}/model_comparison_table_legacy.csv

Usage:
    python analysis/generate_legacy_comparison_table.py --language english
    python analysis/generate_legacy_comparison_table.py --language german
"""

import json
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMPARISONS_DIR = ROOT / "results" / "comparisons"


# ---------------------------------------------------------------------------
# Load per-run metrics.json
# ---------------------------------------------------------------------------

def load_run_metrics(lang: str, model: str, prompt_type: str, seed: int) -> dict:
    path = COMPARISONS_DIR / lang / model / prompt_type / f"seed_{seed}" / "metrics.json"
    if not path.exists():
        print(f"  Warning: metrics.json not found — {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def top_confused_idioms(per_idiom: dict, n: int = 8) -> list[str]:
    """Return top-N idioms by confusion rate, formatted as 'name (confused/total, rate%)'."""
    sorted_idioms = sorted(per_idiom.items(), key=lambda x: x[1]["confusion_rate"], reverse=True)
    result = []
    for idiom, stats in sorted_idioms[:n]:
        confused = stats["confused"]
        total = stats["total"]
        rate = stats["confusion_rate"] * 100
        result.append(f"{idiom} ({confused}/{total}, {rate:.1f}%)")
    return result


def confused_examples(per_sentence: list, n: int = 10) -> list[str]:
    """Extract first N confused variant sentences from per-sentence data."""
    examples = []
    for sent in per_sentence:
        for vd in sent.get("variant_details", []):
            if vd.get("is_confused", False):
                examples.append(vd.get("variant_sentence", ""))
                if len(examples) >= n:
                    return examples
    return examples


# ---------------------------------------------------------------------------
# Build one row in legacy format
# ---------------------------------------------------------------------------

def build_legacy_row(row: pd.Series, lang: str) -> dict:
    model       = row["model"]
    prompt_type = row["prompt_type"]
    seed        = int(row["seed"])
    n           = int(row["matched_sentences"])

    m = load_run_metrics(lang, model, prompt_type, seed)
    dc         = m.get("detailed_confusion", {})
    ov         = dc.get("overall_stats", {})
    lib        = dc.get("literal_idiomatic_breakdown", {})
    per_idiom  = dc.get("per_idiom_breakdown", {})
    per_sent   = dc.get("per_sentence", [])

    ce         = m.get("context_effects", {})
    vls        = m.get("variant_level_stats", {})
    hard_m     = m.get("hard_idioms_metrics", {})
    id10m_m    = m.get("id10m_metrics", {})

    lit_total  = lib.get("literal_total", 0)
    lit_conf   = lib.get("literal_confused", 0)
    idio_total = lib.get("idiomatic_total", 0)
    idio_conf  = lib.get("idiomatic_confused", 0)
    tot_var    = ov.get("total_variants", 0)
    tot_conf_v = ov.get("total_confused_variants", int(row["total_confused_variants"]))

    # id10m-correct sentences only (used as total_analyzed_sentences in old table)
    id10m_correct_sentences = ov.get("total_sentences", n)
    # total variant rows in hard_idioms dataset (not matched sentences)
    hard_total_rows = hard_m.get("total", n)

    top_idioms = top_confused_idioms(per_idiom, n=8)
    examples   = confused_examples(per_sent, n=10)

    result = {
        # Model config
        "model":        model,
        "prompt_type":  prompt_type,
        "shots":        int(row["shots"]),
        "sc_runs":      int(row["sc_runs"]),
        "temperature":  float(row["temperature"]),
        "seed":         seed,
        "language":     row["language"],

        # Overall confusion summary
        # Note: total_analyzed_sentences = id10m-correct sentences only (matches old script)
        "total_analyzed_sentences": id10m_correct_sentences,
        "total_variants":           tot_var,
        "total_confused_variants":  tot_conf_v,
        "total_confusion_percent":  round(float(row["overall_confusion_rate"]) * 100, 2),

        # Literal vs idiomatic breakdown
        "literal_total_variants":      lit_total,
        "literal_confused_variants":   lit_conf,
        "literal_confusion_percent":   round(lit_conf / lit_total * 100, 1) if lit_total else 0.0,
        "idiomatic_total_variants":    idio_total,
        "idiomatic_confused_variants": idio_conf,
        "idiomatic_confusion_percent": round(idio_conf / idio_total * 100, 1) if idio_total else 0.0,

        # Context effects (direct counts from metrics.json, not reconstructed from percentages)
        "context_helped": ce.get("helped", 0),
        "context_hurt":   ce.get("hurt",   0),
        "mixed_results":  ce.get("mixed",  0),
        "no_change":      ce.get("no_change", 0),

        # ID10M performance
        "id10m_total_sentences":    n,
        "id10m_correct_detections": id10m_m.get("correct_detection", 0) + id10m_m.get("correct_rejection", 0),
        "id10m_accuracy":           float(row["id10m_accuracy"]),
        "id10m_precision":          float(row["id10m_precision"]),
        "id10m_recall":             float(row["id10m_recall"]),
        "id10m_f1":                 float(row["id10m_f1"]),
        "id10m_mcc":                float(row["id10m_mcc"]),

        # Hard idioms performance
        # Note: hard_total_sentences = total variant rows (not matched sentences, matches old script)
        "hard_total_sentences":  hard_total_rows,
        "hard_accuracy":         float(row["hard_accuracy"]),
        "hard_precision":        float(row["hard_precision"]),
        "hard_recall":           float(row["hard_recall"]),
        "hard_f1":               float(row["hard_f1"]),
        "hard_mcc":              float(row["hard_mcc"]),
        "hard_false_positives":  int(row["hard_fp"]),
        "hard_false_negatives":  int(row["hard_fn"]),
        "hard_total_errors":     int(row["hard_fp"]) + int(row["hard_fn"]),

        # Advanced context metrics
        "context_degradation_index":   float(row["context_degradation_index"]),
        "variant_consistency_score":   float(row["variant_consistency_score"]),
        "context_confusion_rate":      float(row["context_confusion_rate"]),
        "literal_to_idiom_flip_rate":  float(row["literal_to_idiom_flip_rate"]),
        "error_amplification_factor":  float(row["error_amplification_factor"]),
        "confusion_resistance_score":  float(row["confusion_resistance_score"]),
        "avg_confusion_rate":          float(row["avg_confusion_rate"]),
        # Direct count from variant_level_stats (not reconstructed from percentage)
        "highly_vulnerable_sentences": vls.get("highly_vulnerable", 0),
    }

    # Top-8 confused idioms
    for i in range(1, 9):
        result[f"top_{i}_confused_idiom"] = top_idioms[i - 1] if i <= len(top_idioms) else ""

    # 10 confused variant examples
    for i in range(1, 11):
        result[f"confused_example_{i}"] = examples[i - 1] if i <= len(examples) else ""

    return result


# ---------------------------------------------------------------------------
# Column order matching the old table exactly
# ---------------------------------------------------------------------------

COLUMN_ORDER = [
    "model", "prompt_type", "shots", "sc_runs", "temperature", "seed", "language",
    "total_analyzed_sentences", "total_variants", "total_confused_variants", "total_confusion_percent",
    "literal_total_variants", "literal_confused_variants", "literal_confusion_percent",
    "idiomatic_total_variants", "idiomatic_confused_variants", "idiomatic_confusion_percent",
    "context_helped", "context_hurt", "mixed_results", "no_change",
    "id10m_total_sentences", "id10m_correct_detections",
    "id10m_accuracy", "id10m_precision", "id10m_recall", "id10m_f1", "id10m_mcc",
    "hard_total_sentences",
    "hard_accuracy", "hard_precision", "hard_recall", "hard_f1", "hard_mcc",
    "hard_false_positives", "hard_false_negatives", "hard_total_errors",
    "context_degradation_index", "variant_consistency_score", "context_confusion_rate",
    "literal_to_idiom_flip_rate", "error_amplification_factor", "confusion_resistance_score",
    "avg_confusion_rate", "highly_vulnerable_sentences",
    "top_1_confused_idiom", "top_2_confused_idiom", "top_3_confused_idiom", "top_4_confused_idiom",
    "top_5_confused_idiom", "top_6_confused_idiom", "top_7_confused_idiom", "top_8_confused_idiom",
    "confused_example_1", "confused_example_2", "confused_example_3", "confused_example_4",
    "confused_example_5", "confused_example_6", "confused_example_7", "confused_example_8",
    "confused_example_9", "confused_example_10",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="english", choices=["english", "german"])
    args = parser.parse_args()
    lang = args.language

    src_csv    = COMPARISONS_DIR / lang / "model_comparison_table.csv"
    output_csv = COMPARISONS_DIR / lang / "model_comparison_table_legacy.csv"

    print("=" * 70)
    print(f"Generating legacy comparison table — {lang}")
    print("=" * 70)

    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df)} rows from {src_csv.name}\n")

    rows = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        row_dict = row._asdict()
        row_series = pd.Series(row_dict)
        print(f"[{i}/{len(df)}] {row.model} / {row.prompt_type} / seed {row.seed}")
        rows.append(build_legacy_row(row_series, lang))

    out_df = pd.DataFrame(rows, columns=COLUMN_ORDER)
    out_df = out_df.sort_values("hard_f1", ascending=False)
    out_df.to_csv(output_csv, index=False)

    print(f"\n✓ Written {len(out_df)} rows → {output_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
