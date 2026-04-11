#!/usr/bin/env python3
"""
Sentence-Level Confusion Analysis Script

For each original sentence (grouped across all its hard_idioms variants), determines
whether ALL variants confused the model, NO variants confused the model, or results
were MIXED.

Output CSV columns per model config:
  all_confused_count / percent  — sentences where every variant confused the model
  none_confused_count / percent — sentences where no variant confused the model
  mixed_count / mixed_percent   — sentences with mixed results across variants
  + 3 example texts per category

Usage:
    python analysis/sentence_confusion_analysis.py --language english
    python analysis/sentence_confusion_analysis.py --language german
    python analysis/sentence_confusion_analysis.py --language english --output custom.csv
"""

import json
import csv
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HARD_DIR = ROOT / "results" / "hard_idioms"
COMPARISON_DIR = ROOT / "results" / "comparisons"


# ---------------------------------------------------------------------------
# List coercion (German data stores true_idioms as Python-repr strings)
# ---------------------------------------------------------------------------

def _coerce_list(val) -> List[str]:
    """Convert a value to a list of strings.
    Handles the German hard_idioms artifact where true_idioms is stored as
    a Python repr string like \"['mitgehen lassen']\" instead of a real list.
    """
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val:
        return []
    s = val.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return [s]
    tokens = re.findall(r"['\"]([^'\"]*)['\"]", s)
    return tokens if tokens else []


# ---------------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------------

def _get_idioms_from_response(r: dict) -> List[str]:
    """Extract idioms list from a single response entry."""
    parsed = r.get("parsed", {})
    if isinstance(parsed, dict):
        idioms = parsed.get("idioms", [])
    elif isinstance(parsed, list):
        idioms = parsed
    else:
        return []
    if isinstance(idioms, list):
        return [str(i) for i in idioms if i is not None]
    if isinstance(idioms, str):
        return [idioms] if idioms else []
    return []


def get_predicted_idioms(item: dict) -> List[str]:
    """
    Extract predicted idioms using majority vote across SC responses.
    For single-response runs (sc_runs=1), returns that response's idioms.
    For SC runs (sc_runs>1), an idiom is accepted if it appears in >50% of runs.
    """
    responses = item.get("responses", [])
    if not responses:
        return []

    if len(responses) == 1:
        return _get_idioms_from_response(responses[0])

    # Majority vote: collect all idiom strings, keep those appearing > len/2 times
    n = len(responses)
    counts: Dict[str, int] = defaultdict(int)
    for r in responses:
        for idiom in _get_idioms_from_response(r):
            counts[idiom.lower().strip()] += 1
    return [idiom for idiom, cnt in counts.items() if cnt > n / 2]


# ---------------------------------------------------------------------------
# Confusion classification
# ---------------------------------------------------------------------------

def analyze_confusion(true_idioms: List[str], predicted_idioms: List[str]) -> str:
    """
    Classify a single variant prediction.
    Uses bidirectional substring matching (lenient), same as the old script.

    Returns one of:
        'correct_detection', 'correct_rejection', 'false_positive', 'false_negative'
    """
    has_true = bool(true_idioms)
    has_pred = bool(predicted_idioms)

    if not has_true and not has_pred:
        return "correct_rejection"
    if not has_true and has_pred:
        return "false_positive"
    if has_true and not has_pred:
        return "false_negative"

    # Both non-empty — check substring match for every true idiom
    true_norm = [t.lower().strip() for t in true_idioms]
    pred_norm = [p.lower().strip() for p in predicted_idioms]
    for t in true_norm:
        if not any(t in p or p in t for p in pred_norm):
            return "false_negative"
    return "correct_detection"


def is_confused(conf_type: str) -> bool:
    return conf_type in ("false_positive", "false_negative")


def classify_sentence(variants: List[dict]) -> Tuple[str, List[bool]]:
    """
    Given all variant dicts for one original sentence, return:
        classification : "ALL_CONFUSED" | "NONE_CONFUSED" | "MIXED"
        statuses       : list of bool (True = confused)
    """
    statuses = [
        is_confused(analyze_confusion(v["true_idioms"], v["predicted_idioms"]))
        for v in variants
    ]
    confused = sum(statuses)
    total = len(statuses)
    if confused == 0:
        return "NONE_CONFUSED", statuses
    if confused == total:
        return "ALL_CONFUSED", statuses
    return "MIXED", statuses


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_by_sentence(responses: List[dict]) -> Dict[str, List[dict]]:
    """
    Group response items by their original sentence.
    Each group entry is {'variant_sentence': ..., 'true_idioms': ..., 'predicted_idioms': ...}.
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for item in responses:
        sentence = item.get("sentence", "")
        if not sentence:
            continue
        groups[sentence].append({
            "variant_sentence": item.get("variant_sentence", ""),
            "true_idioms": _coerce_list(item.get("true_idioms", [])),
            "predicted_idioms": get_predicted_idioms(item),
        })
    return dict(groups)


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------

def select_examples(
    classifications: Dict[str, Tuple[str, List[dict]]],
    category: str,
    max_examples: int = 3,
) -> List[str]:
    """
    Return up to max_examples JSON strings {"sentence": ..., "variant_sentence": ...}
    for the given category. Padded with "" if fewer examples exist.
    """
    examples = []
    for sentence, (cls, variants) in classifications.items():
        if cls == category and variants:
            examples.append(json.dumps(
                {"sentence": sentence, "variant_sentence": variants[0]["variant_sentence"]},
                ensure_ascii=False,
            ))
            if len(examples) >= max_examples:
                break
    while len(examples) < max_examples:
        examples.append("")
    return examples[:max_examples]


# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------

def process_model(model: str, prompt_type: str, seed: int, language: str) -> dict | None:
    """
    Load responses for one model config and return the analysis row dict.
    Path: results/hard_idioms/{lang}/updated/{model}/{prompt_type}/seed_{seed}/responses.json
    """
    run_dir = HARD_DIR / language / "updated" / model / prompt_type / f"seed_{seed}"
    responses_path = run_dir / "responses.json"

    if not responses_path.exists():
        print(f"  Warning: not found — {responses_path}")
        return None

    with open(responses_path, encoding="utf-8-sig") as f:
        responses = json.load(f)

    sentence_groups = group_by_sentence(responses)

    classifications: Dict[str, Tuple[str, List[dict]]] = {}
    for sentence, variants in sentence_groups.items():
        cls, statuses = classify_sentence(variants)
        classifications[sentence] = (cls, variants)

    all_confused = sum(1 for c, _ in classifications.values() if c == "ALL_CONFUSED")
    none_confused = sum(1 for c, _ in classifications.values() if c == "NONE_CONFUSED")
    mixed = sum(1 for c, _ in classifications.values() if c == "MIXED")
    total = len(classifications)

    pct = lambda n: f"{n/total*100:.1f}" if total else "0.0"

    all_ex   = select_examples(classifications, "ALL_CONFUSED")
    none_ex  = select_examples(classifications, "NONE_CONFUSED")
    mixed_ex = select_examples(classifications, "MIXED")

    print(
        f"  ✓ {model} / {prompt_type} / seed {seed} — "
        f"{total} sentences: {all_confused} all-confused, {none_confused} none-confused, {mixed} mixed"
    )

    return {
        "model": model, "prompt_type": prompt_type, "seed": seed, "language": language,
        "total_analyzed_sentences": total,
        "all_confused_count": all_confused,
        "all_confused_percent": pct(all_confused),
        "none_confused_count": none_confused,
        "none_confused_percent": pct(none_confused),
        "mixed_count": mixed,
        "mixed_percent": pct(mixed),
        "all_confused_example_1": all_ex[0],
        "all_confused_example_2": all_ex[1],
        "all_confused_example_3": all_ex[2],
        "none_confused_example_1": none_ex[0],
        "none_confused_example_2": none_ex[1],
        "none_confused_example_3": none_ex[2],
        "mixed_example_1": mixed_ex[0],
        "mixed_example_2": mixed_ex[1],
        "mixed_example_3": mixed_ex[2],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "model", "prompt_type", "seed", "language",
    "total_analyzed_sentences",
    "all_confused_count", "all_confused_percent",
    "none_confused_count", "none_confused_percent",
    "mixed_count", "mixed_percent",
    "all_confused_example_1", "all_confused_example_2", "all_confused_example_3",
    "none_confused_example_1", "none_confused_example_2", "none_confused_example_3",
    "mixed_example_1", "mixed_example_2", "mixed_example_3",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="english", choices=["english", "german"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    lang = args.language
    comparison_csv = COMPARISON_DIR / lang / "model_comparison_table.csv"
    output_path = Path(args.output) if args.output else COMPARISON_DIR / lang / "sentence_confusion_table.csv"

    print("=" * 70)
    print(f"Sentence-Level Confusion Analysis — {lang}")
    print("=" * 70)

    if not comparison_csv.exists():
        print(f"ERROR: {comparison_csv} not found")
        return

    df = pd.read_csv(comparison_csv)
    print(f"Loaded {len(df)} model configs from {comparison_csv.name}\n")

    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        print(f"[{i}/{len(df)}] {row.model} / {row.prompt_type} / seed {row.seed}")
        result = process_model(row.model, row.prompt_type, int(row.seed), lang)
        if result:
            rows.append(result)

    print()
    if not rows:
        print("ERROR: no models processed successfully")
        return

    rows.sort(key=lambda r: (r["model"], r["prompt_type"], r["seed"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Written {len(rows)} rows → {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
