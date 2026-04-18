#!/usr/bin/env python3
"""
Generate encoder comparison tables from Alon's new_results.json runs.

Reads new_results.json (keyed by sentence → [{word, tag}]) for updated google_bert
models, cross-references with hard_idioms FINAL.json as ground truth, and writes:
  results/encoders/{language}/encoder_comparison_table.csv
  results/encoders/{language}/encoder_sentence_confusion_table.csv

Only entries present in FINAL.json are used — invalid sentences are excluded
automatically.

Usage:
    python scripts/generate_encoder_comparison_tables.py
    python scripts/generate_encoder_comparison_tables.py --language english
    python scripts/generate_encoder_comparison_tables.py --language german
"""
import ast
import csv
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent

ENCODERS_BASE = Path("/Users/liorlivyatan/Desktop/Livyatan/Thesis/hard_idioms/encoders")

UPDATED_MODELS = {
    "english": [
        "google_bert_bert_base_uncased",
        "google_bert_bert_base_multilingual_cased",
    ],
    "german": [
        "google_bert_bert_base_german_cased",
        "google_bert_bert_base_multilingual_cased",
    ],
}

FINAL_JSON_PATH = {
    "english": ROOT / "data" / "hard_idioms_data" / "english" / "hard_idioms_english_FINAL.json",
    "german": ROOT / "data" / "hard_idioms_data" / "german" / "hard_idioms_german_FINAL.json",
}

SEEDS = [5, 7, 42, 123, 1773]

# ── BIO extraction ────────────────────────────────────────────────────────────

def extract_idiom_span(predictions: List[Dict]) -> List[str]:
    """Extract idiom span strings from a list of {word, tag} BIO predictions.

    Treats 'undefined' as I-IDIOM when already inside a span — this tag appears
    in German predictions for words with attached punctuation at sentence end.
    """
    idioms, cur_words, inside = [], [], False
    for item in predictions:
        tag = item.get("tag", "O")
        word = item.get("word", "")
        if tag == "B-IDIOM":
            if cur_words:
                idioms.append(" ".join(cur_words))
            cur_words = [word]
            inside = True
        elif (tag == "I-IDIOM" or (tag == "undefined" and inside)):
            if inside:
                cur_words.append(word)
        else:
            if cur_words:
                idioms.append(" ".join(cur_words))
                cur_words = []
            inside = False
    if cur_words:
        idioms.append(" ".join(cur_words))
    return idioms


# ── Data loading ──────────────────────────────────────────────────────────────

def load_final_json(language: str) -> List[Dict]:
    with open(FINAL_JSON_PATH[language], encoding="utf-8") as f:
        return json.load(f)


def load_new_results(language: str, model: str, seed: int) -> Optional[Dict]:
    p = ENCODERS_BASE / language / model / str(seed) / "results" / "new_results.json"
    if not p.exists():
        print(f"  SKIP: {p} not found")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse_true_idioms(raw) -> List[str]:
    """Parse true_idioms — English FINAL.json stores them as lists, German as strings."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        # ast.literal_eval is safe: only parses Python literals, never executes code
        return ast.literal_eval(raw)
    return []


def build_comparison_data(final_data: List[Dict], new_results: Dict) -> List[Dict]:
    """
    Join FINAL.json with new_results.json predictions.
    Only includes entries where both variant_sentence and sentence are in new_results.
    Invalid sentences (not in FINAL.json) are automatically excluded.
    """
    rows = []
    for entry in final_data:
        orig = entry["sentence"]
        variant = entry["variant_sentence"]
        true_idioms = parse_true_idioms(entry["true_idioms"])

        if variant not in new_results or orig not in new_results:
            continue

        rows.append({
            "final_variant": variant,
            "orig_sentence": orig,
            "true_idioms": true_idioms,
            "variant_predicted_idioms": extract_idiom_span(new_results[variant]),
            "orig_sentence_idioms": extract_idiom_span(new_results[orig]),
        })
    return rows


# ── Confusion metrics ─────────────────────────────────────────────────────────

def analyze_confusion(true_idioms: List[str], predicted_idioms: List[str]) -> str:
    """Classify prediction using lenient bidirectional substring matching."""
    has_true = len(true_idioms) > 0
    has_pred = len(predicted_idioms) > 0

    if not has_true and not has_pred:
        return "correct_rejection"
    elif not has_true and has_pred:
        return "false_positive"
    elif has_true and not has_pred:
        return "false_negative"
    else:
        true_norm = [t.lower().strip() for t in true_idioms]
        pred_norm = [p.lower().strip() for p in predicted_idioms]
        for true_idiom in true_norm:
            if not any(true_idiom in p or p in true_idiom for p in pred_norm):
                return "false_negative"
        return "correct_detection"


@dataclass
class ConfusionMetrics:
    correct_detection: int = 0
    correct_rejection: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def total(self):
        return self.correct_detection + self.correct_rejection + self.false_positive + self.false_negative

    @property
    def accuracy(self):
        return (self.correct_detection + self.correct_rejection) / self.total if self.total else 0.0

    @property
    def precision(self):
        denom = self.correct_detection + self.false_positive
        return self.correct_detection / denom if denom else 0.0

    @property
    def recall(self):
        denom = self.correct_detection + self.false_negative
        return self.correct_detection / denom if denom else 0.0

    @property
    def f1_score(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mcc(self):
        tp = self.correct_detection
        tn = self.correct_rejection
        fp = self.false_positive
        fn = self.false_negative
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return ((tp * tn) - (fp * fn)) / denom if denom else 0.0


def calc_metrics(data: List[Dict], use_orig: bool) -> ConfusionMetrics:
    m = ConfusionMetrics()
    key = "orig_sentence_idioms" if use_orig else "variant_predicted_idioms"
    for item in data:
        ct = analyze_confusion(item["true_idioms"], item[key])
        if ct == "correct_detection":
            m.correct_detection += 1
        elif ct == "correct_rejection":
            m.correct_rejection += 1
        elif ct == "false_positive":
            m.false_positive += 1
        else:
            m.false_negative += 1
    return m


def context_effects(data: List[Dict]) -> Dict:
    helped = hurt = no_change = 0
    for item in data:
        orig_ok = analyze_confusion(item["true_idioms"], item["orig_sentence_idioms"]) in (
            "correct_detection", "correct_rejection"
        )
        var_ok = analyze_confusion(item["true_idioms"], item["variant_predicted_idioms"]) in (
            "correct_detection", "correct_rejection"
        )
        if not orig_ok and var_ok:
            helped += 1
        elif orig_ok and not var_ok:
            hurt += 1
        else:
            no_change += 1
    return {"context_helped": helped, "context_hurt": hurt, "mixed_results": 0, "no_change": no_change}


def confusion_breakdown(data: List[Dict]) -> Dict:
    lit_total = lit_conf = idio_total = idio_conf = 0
    for item in data:
        is_lit = len(item["true_idioms"]) == 0
        is_confused = analyze_confusion(
            item["true_idioms"], item["variant_predicted_idioms"]
        ) not in ("correct_detection", "correct_rejection")
        if is_lit:
            lit_total += 1
            lit_conf += is_confused
        else:
            idio_total += 1
            idio_conf += is_confused
    return {
        "literal_total_variants": lit_total,
        "literal_confused_variants": lit_conf,
        "literal_confusion_percent": lit_conf / lit_total * 100 if lit_total else 0.0,
        "idiomatic_total_variants": idio_total,
        "idiomatic_confused_variants": idio_conf,
        "idiomatic_confusion_percent": idio_conf / idio_total * 100 if idio_total else 0.0,
    }


def advanced_metrics(id10m: ConfusionMetrics, hard: ConfusionMetrics, data: List[Dict]) -> Dict:
    cdi = (id10m.accuracy - hard.accuracy) / id10m.accuracy if id10m.accuracy else 0.0

    ccr_count = total_count = 0
    lifr_count = lit_count = 0
    crs_scores = []

    for item in data:
        ti = item["true_idioms"]
        orig_ok = analyze_confusion(ti, item["orig_sentence_idioms"]) in (
            "correct_detection", "correct_rejection"
        )
        var_ok = analyze_confusion(ti, item["variant_predicted_idioms"]) in (
            "correct_detection", "correct_rejection"
        )
        total_count += 1
        if orig_ok and not var_ok:
            ccr_count += 1
        if orig_ok:
            crs_scores.append(0 if var_ok else 1)
        if len(ti) == 0 and orig_ok:
            lit_count += 1
            if not var_ok:
                lifr_count += 1

    id10m_err = 1 - id10m.accuracy
    hard_err = 1 - hard.accuracy
    eaf = hard_err / id10m_err if id10m_err else 1.0

    confused_total = sum(
        1 for item in data
        if analyze_confusion(item["true_idioms"], item["variant_predicted_idioms"])
        not in ("correct_detection", "correct_rejection")
    )

    return {
        "context_degradation_index": cdi,
        "variant_consistency_score": 1.0,
        "context_confusion_rate": ccr_count / total_count if total_count else 0.0,
        "literal_to_idiom_flip_rate": lifr_count / lit_count if lit_count else 0.0,
        "error_amplification_factor": eaf,
        "confusion_resistance_score": sum(crs_scores) / len(crs_scores) if crs_scores else 0.0,
        "avg_confusion_rate": confused_total / len(data) if data else 0.0,
        "highly_vulnerable_sentences": confused_total,
    }


def top_confused_idioms(data: List[Dict], n: int = 10) -> List[str]:
    stats = defaultdict(lambda: {"total": 0, "confused": 0})
    for item in data:
        if item["true_idioms"]:
            is_confused = analyze_confusion(
                item["true_idioms"], item["variant_predicted_idioms"]
            ) not in ("correct_detection", "correct_rejection")
            for idiom in item["true_idioms"]:
                stats[idiom]["total"] += 1
                stats[idiom]["confused"] += is_confused

    ranked = sorted(
        [(i, s["confused"], s["total"], s["confused"] / s["total"] * 100)
         for i, s in stats.items() if s["total"] > 0],
        key=lambda x: (x[3], x[2]), reverse=True,
    )
    results = [f"{i} ({c}/{t}, {r:.1f}%)" for i, c, t, r in ranked[:n]]
    while len(results) < n:
        results.append("")
    return results


def confused_examples(data: List[Dict], n: int = 10) -> List[str]:
    ex = []
    for item in data:
        if len(ex) >= n:
            break
        if analyze_confusion(item["true_idioms"], item["variant_predicted_idioms"]) not in (
            "correct_detection", "correct_rejection"
        ):
            ex.append(item["final_variant"])
    while len(ex) < n:
        ex.append("")
    return ex


def convert_model_name(dir_name: str) -> str:
    parts = dir_name.split("_")
    return f"{parts[0]}/{'-'.join(parts[1:])}" if len(parts) >= 2 else dir_name


# ── Comparison table ──────────────────────────────────────────────────────────

COMPARISON_FIELDNAMES = [
    "model", "seed", "language",
    "total_analyzed_sentences", "total_variants", "total_confused_variants", "total_confusion_percent",
    "literal_total_variants", "literal_confused_variants", "literal_confusion_percent",
    "idiomatic_total_variants", "idiomatic_confused_variants", "idiomatic_confusion_percent",
    "context_helped", "context_hurt", "mixed_results", "no_change",
    "id10m_total_sentences", "id10m_accuracy", "id10m_precision", "id10m_recall", "id10m_f1", "id10m_mcc",
    "hard_total_sentences", "hard_accuracy", "hard_precision", "hard_recall", "hard_f1", "hard_mcc",
    "context_degradation_index", "variant_consistency_score", "context_confusion_rate",
    "literal_to_idiom_flip_rate", "error_amplification_factor", "confusion_resistance_score",
    "avg_confusion_rate", "highly_vulnerable_sentences",
    "top_1_confused_idiom", "top_2_confused_idiom", "top_3_confused_idiom", "top_4_confused_idiom",
    "top_5_confused_idiom", "top_6_confused_idiom", "top_7_confused_idiom", "top_8_confused_idiom",
    "top_9_confused_idiom", "top_10_confused_idiom",
    "confused_example_1", "confused_example_2", "confused_example_3", "confused_example_4",
    "confused_example_5", "confused_example_6", "confused_example_7", "confused_example_8",
    "confused_example_9", "confused_example_10",
]


def process_model_comparison(data: List[Dict], model: str, seed: int, language: str) -> Dict:
    id10m = calc_metrics(data, use_orig=True)
    hard = calc_metrics(data, use_orig=False)
    ctx = context_effects(data)
    cb = confusion_breakdown(data)
    adv = advanced_metrics(id10m, hard, data)
    top = top_confused_idioms(data)
    ex = confused_examples(data)

    total_variants = len(data)
    total_analyzed = len(set(item["orig_sentence"] for item in data))
    confused = adv["highly_vulnerable_sentences"]

    return {
        "model": convert_model_name(model),
        "seed": seed,
        "language": language,
        "total_analyzed_sentences": total_analyzed,
        "total_variants": total_variants,
        "total_confused_variants": confused,
        "total_confusion_percent": round(confused / total_variants * 100, 1) if total_variants else 0.0,
        "literal_total_variants": cb["literal_total_variants"],
        "literal_confused_variants": cb["literal_confused_variants"],
        "literal_confusion_percent": round(cb["literal_confusion_percent"], 1),
        "idiomatic_total_variants": cb["idiomatic_total_variants"],
        "idiomatic_confused_variants": cb["idiomatic_confused_variants"],
        "idiomatic_confusion_percent": round(cb["idiomatic_confusion_percent"], 1),
        "context_helped": ctx["context_helped"],
        "context_hurt": ctx["context_hurt"],
        "mixed_results": 0,
        "no_change": ctx["no_change"],
        "id10m_total_sentences": id10m.total,
        "id10m_accuracy": round(id10m.accuracy, 3),
        "id10m_precision": round(id10m.precision, 3),
        "id10m_recall": round(id10m.recall, 3),
        "id10m_f1": round(id10m.f1_score, 3),
        "id10m_mcc": round(id10m.mcc, 3),
        "hard_total_sentences": hard.total,
        "hard_accuracy": round(hard.accuracy, 3),
        "hard_precision": round(hard.precision, 3),
        "hard_recall": round(hard.recall, 3),
        "hard_f1": round(hard.f1_score, 3),
        "hard_mcc": round(hard.mcc, 3),
        **{k: round(v, 3) if isinstance(v, float) else v for k, v in adv.items()},
        **{f"top_{i+1}_confused_idiom": top[i] for i in range(10)},
        **{f"confused_example_{i+1}": ex[i] for i in range(10)},
    }


# ── Sentence confusion table ──────────────────────────────────────────────────

SENTENCE_CONFUSION_FIELDNAMES = [
    "model", "seed", "language",
    "all_confused_count", "all_confused_percent",
    "none_confused_count", "none_confused_percent",
    "mixed_count", "mixed_percent",
    "all_confused_example_1", "all_confused_example_2", "all_confused_example_3",
    "none_confused_example_1", "none_confused_example_2", "none_confused_example_3",
    "mixed_example_1", "mixed_example_2", "mixed_example_3",
]


def classify_sentence_confusion(item: Dict) -> str:
    """ALL_CONFUSED if variant is confused, else NONE_CONFUSED."""
    ct = analyze_confusion(item["true_idioms"], item["variant_predicted_idioms"])
    return "ALL_CONFUSED" if ct in ("false_positive", "false_negative") else "NONE_CONFUSED"


def filter_orig_correct(data: List[Dict]) -> List[Dict]:
    """Keep only sentences where original prediction was correct."""
    return [
        item for item in data
        if analyze_confusion(item["true_idioms"], item["orig_sentence_idioms"])
        in ("correct_detection", "correct_rejection")
    ]


def select_sentence_examples(data: List[Dict], category: str, n: int = 3) -> List[str]:
    ex = []
    for item in data:
        if len(ex) >= n:
            break
        if classify_sentence_confusion(item) == category:
            ex.append(json.dumps(
                {"sentence": item["orig_sentence"], "final_variant": item["final_variant"]},
                ensure_ascii=False,
            ))
    while len(ex) < n:
        ex.append("")
    return ex


def process_model_sentence_confusion(data: List[Dict], model: str, seed: int, language: str) -> Dict:
    filtered = filter_orig_correct(data)
    if not filtered:
        return None

    all_conf = sum(1 for item in filtered if classify_sentence_confusion(item) == "ALL_CONFUSED")
    none_conf = len(filtered) - all_conf
    total = len(filtered)

    return {
        "model": convert_model_name(model),
        "seed": seed,
        "language": language,
        "all_confused_count": all_conf,
        "all_confused_percent": f"{all_conf/total*100:.1f}" if total else "0.0",
        "none_confused_count": none_conf,
        "none_confused_percent": f"{none_conf/total*100:.1f}" if total else "0.0",
        "mixed_count": 0,
        "mixed_percent": "0.0",
        "all_confused_example_1": select_sentence_examples(filtered, "ALL_CONFUSED")[0],
        "all_confused_example_2": select_sentence_examples(filtered, "ALL_CONFUSED")[1],
        "all_confused_example_3": select_sentence_examples(filtered, "ALL_CONFUSED")[2],
        "none_confused_example_1": select_sentence_examples(filtered, "NONE_CONFUSED")[0],
        "none_confused_example_2": select_sentence_examples(filtered, "NONE_CONFUSED")[1],
        "none_confused_example_3": select_sentence_examples(filtered, "NONE_CONFUSED")[2],
        "mixed_example_1": "", "mixed_example_2": "", "mixed_example_3": "",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_language(language: str) -> None:
    print(f"\n{'='*60}")
    print(f"Language: {language}")
    print(f"{'='*60}")

    final_data = load_final_json(language)
    print(f"Loaded FINAL.json: {len(final_data)} valid entries")

    out_dir = ROOT / "results" / "encoders" / language
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    sentence_confusion_rows = []

    models = UPDATED_MODELS[language]
    total = len(models) * len(SEEDS)
    current = 0

    for model in models:
        for seed in SEEDS:
            current += 1
            print(f"\n[{current}/{total}] {model} / seed {seed}")

            new_results = load_new_results(language, model, seed)
            if new_results is None:
                continue

            data = build_comparison_data(final_data, new_results)
            if not data:
                print("  No valid data after join — skipping")
                continue

            print(f"  Built {len(data)} entries ({len(set(d['orig_sentence'] for d in data))} unique originals)")

            row_comp = process_model_comparison(data, model, seed, language)
            comparison_rows.append(row_comp)
            print(f"  id10m F1={row_comp['id10m_f1']:.3f}  hard F1={row_comp['hard_f1']:.3f}  confusion={row_comp['total_confusion_percent']}%")

            row_sent = process_model_sentence_confusion(data, model, seed, language)
            if row_sent:
                sentence_confusion_rows.append(row_sent)

    # Write comparison table
    if comparison_rows:
        comparison_rows.sort(key=lambda x: (x["model"], x["seed"]))
        comp_path = out_dir / "encoder_comparison_table.csv"
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDNAMES)
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"\nWrote {len(comparison_rows)} rows → {comp_path}")

    # Write sentence confusion table
    if sentence_confusion_rows:
        sentence_confusion_rows.sort(key=lambda x: (x["model"], x["seed"]))
        sent_path = out_dir / "encoder_sentence_confusion_table.csv"
        with open(sent_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SENTENCE_CONFUSION_FIELDNAMES)
            writer.writeheader()
            writer.writerows(sentence_confusion_rows)
        print(f"Wrote {len(sentence_confusion_rows)} rows → {sent_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["english", "german"], default=None,
                        help="Language to process (default: both)")
    args = parser.parse_args()

    langs = [args.language] if args.language else ["english", "german"]
    for lang in langs:
        run_language(lang)
    print("\nDone.")


if __name__ == "__main__":
    main()
