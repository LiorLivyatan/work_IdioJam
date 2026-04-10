"""
Batch comparison: id10m vs hard_idioms (updated FINAL dataset).

For each model/prompt/seed that exists in BOTH result sets, computes:
  - id10m metrics on the matched sentence subset
  - hard_idioms metrics on the same sentences' variants
  - Advanced context-confusion metrics (CDI, VCS, CCR, EAF, etc.)

Outputs per-run:
  results/comparisons/{lang}/{model}/{prompt}/seed_{N}/
      comparison_report.txt
      metrics.json

Outputs aggregated:
  results/comparisons/{lang}/model_comparison_table.csv

Usage:
    python analysis/compare_id10m_vs_hard_idioms.py                     # all langs
    python analysis/compare_id10m_vs_hard_idioms.py --lang english
    python analysis/compare_id10m_vs_hard_idioms.py --lang german
    python analysis/compare_id10m_vs_hard_idioms.py --dry_run
    python analysis/compare_id10m_vs_hard_idioms.py --filter gpt-4o-mini
"""

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

ID10M_DIR      = REPO_ROOT / "results" / "id10m"
HARD_DIR       = REPO_ROOT / "results" / "hard_idioms"
COMPARISONS_DIR = REPO_ROOT / "results" / "comparisons"

LANGUAGES = ["english", "german"]

####################################################################################################
# Data structures

@dataclass
class SentenceData:
    sentence: str
    normalized_sentence: str
    true_idioms: List[str]
    predicted_idioms: List[str]
    source_dataset: str
    variant_number: Optional[int] = None
    variant_sentence: Optional[str] = None   # replaces old 'final_variant'


@dataclass
class ConfusionMetrics:
    correct_detection: int = 0
    correct_rejection: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def total(self): return self.correct_detection + self.correct_rejection + self.false_positive + self.false_negative

    @property
    def accuracy(self):
        return (self.correct_detection + self.correct_rejection) / self.total if self.total else 0.0

    @property
    def precision(self):
        pp = self.correct_detection + self.false_positive
        return self.correct_detection / pp if pp else 0.0

    @property
    def recall(self):
        ap = self.correct_detection + self.false_negative
        return self.correct_detection / ap if ap else 0.0

    @property
    def specificity(self):
        an = self.correct_rejection + self.false_positive
        return self.correct_rejection / an if an else 0.0

    @property
    def f1_score(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def balanced_accuracy(self):
        return (self.recall + self.specificity) / 2

    @property
    def mcc(self):
        tp, tn, fp, fn = self.correct_detection, self.correct_rejection, self.false_positive, self.false_negative
        denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return ((tp*tn) - (fp*fn)) / denom if denom else 0.0


@dataclass
class AdvancedMetrics:
    context_degradation_index: float = 0.0
    variant_consistency_score: float = 0.0
    context_confusion_rate: float = 0.0
    literal_to_idiom_flip_rate: float = 0.0
    error_amplification_factor: float = 0.0
    phrase_overlap_confusions: float = 0.0
    confusion_resistance_score: float = 0.0
    prediction_stability_scores: List[float] = field(default_factory=list)


####################################################################################################
# Parsing helpers

def normalize_sentence(s: str) -> str:
    """Normalize sentence for matching across datasets."""
    s = re.sub(r'\s*―\s*.*$', '', s)           # remove English translations
    s = re.sub(r'\s*\([^)]*\)\s*$', '', s)      # remove dialect metadata
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\s+([.!?,:;])', r'\1', s)
    return s


def _extract_predicted_idioms(item: dict) -> List[str]:
    """Extract predicted idiom list from a responses.json entry.
    Handles two formats:
      - New: responses[0].parsed.idioms  (hard_idioms + newer id10m runs)
      - Old: responses[0].idioms         (older id10m runs)
    """
    def _parse_idioms(idioms):
        if isinstance(idioms, list):
            return [str(i) for i in idioms if i is not None]
        if isinstance(idioms, str):
            return [idioms] if idioms else []
        return []

    try:
        responses = item.get("responses", [])
        if not responses:
            return []
        resp = responses[0]

        # New format: parsed.idioms
        parsed = resp.get("parsed")
        if isinstance(parsed, dict):
            idioms = parsed.get("idioms", [])
            if idioms is not None:
                return _parse_idioms(idioms)
        elif isinstance(parsed, list):
            return _parse_idioms(parsed)

        # Old format: idioms directly on response
        if "idioms" in resp:
            return _parse_idioms(resp["idioms"])

    except Exception:
        pass
    return []


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_id10m(path: Path) -> List[SentenceData]:
    data = load_json(path)
    out = []
    for item in data:
        out.append(SentenceData(
            sentence=item["sentence"],
            normalized_sentence=normalize_sentence(item["sentence"]),
            true_idioms=item.get("true_idioms", []),
            predicted_idioms=_extract_predicted_idioms(item),
            source_dataset="id10m",
        ))
    return out


def load_hard_idioms(path: Path) -> List[SentenceData]:
    data = load_json(path)
    out = []
    for item in data:
        out.append(SentenceData(
            sentence=item["sentence"],
            normalized_sentence=normalize_sentence(item["sentence"]),
            true_idioms=item.get("true_idioms", []),
            predicted_idioms=_extract_predicted_idioms(item),
            source_dataset="hard_idioms",
            variant_number=item.get("variant_number"),
            variant_sentence=item.get("variant_sentence"),   # new field name
        ))
    return out


def find_matches(id10m_sents: List[SentenceData],
                 hard_sents: List[SentenceData]) -> Dict[str, dict]:
    """Return dict keyed by normalized sentence with id10m + hard_idioms variants."""
    id10m_lut = {s.normalized_sentence: s for s in id10m_sents}
    hard_lut: Dict[str, List[SentenceData]] = defaultdict(list)
    for s in hard_sents:
        hard_lut[s.normalized_sentence].append(s)

    matches = {}
    for norm, sent in id10m_lut.items():
        if norm in hard_lut:
            matches[norm] = {"id10m": sent, "hard_idioms": hard_lut[norm]}
    return matches


####################################################################################################
# Confusion analysis

def analyze_confusion(true_idioms: List[str], predicted_idioms: List[str]) -> str:
    """Lenient substring matching in both directions."""
    has_true = len(true_idioms) > 0
    has_pred = len(predicted_idioms) > 0
    if not has_true and not has_pred:
        return "correct_rejection"
    if not has_true and has_pred:
        return "false_positive"
    if has_true and not has_pred:
        return "false_negative"
    t_norm = [t.lower().strip() for t in true_idioms]
    p_norm = [p.lower().strip() for p in predicted_idioms]
    for t in t_norm:
        if not any(t in p or p in t for p in p_norm):
            return "false_negative"
    return "correct_detection"


def _update_metrics(m: ConfusionMetrics, ctype: str):
    if ctype == "correct_detection":   m.correct_detection += 1
    elif ctype == "correct_rejection": m.correct_rejection += 1
    elif ctype == "false_positive":    m.false_positive += 1
    elif ctype == "false_negative":    m.false_negative += 1


def analyze_context_effect(id10m_ct: str, hard_cts: List[str]) -> str:
    id10m_ok = id10m_ct in ("correct_detection", "correct_rejection")
    hard_ok = [c in ("correct_detection", "correct_rejection") for c in hard_cts]
    if all(hard_ok) and not id10m_ok:  return "helped"
    if not any(hard_ok) and id10m_ok:  return "hurt"
    if len(set(hard_ok)) > 1:          return "mixed"
    return "no_change"


####################################################################################################
# Advanced metrics calculators

def _vcs(variant_predictions: List[List[str]]) -> float:
    if len(variant_predictions) <= 1:
        return 1.0
    binary = [1 if p else 0 for p in variant_predictions]
    if len(set(binary)) == 1:
        return 1.0
    return 1 - (np.var(binary) / 0.25)


def _psi(variant_predictions: List[List[str]]) -> float:
    if len(variant_predictions) <= 1:
        return 0.0
    return float(np.std([len(p) for p in variant_predictions]))


####################################################################################################
# Core comparison

def compare(matches: Dict[str, dict]) -> dict:
    results = {
        "sentence_comparisons": [],
        "id10m_metrics": ConfusionMetrics(),
        "hard_idioms_metrics": ConfusionMetrics(),
        "context_effects": {"helped": [], "hurt": [], "no_change": [], "mixed": []},
        "vcs_scores": [],
        "psi_scores": [],
    }

    for norm, data in matches.items():
        id10m_s = data["id10m"]
        variants = data["hard_idioms"]

        id10m_ct = analyze_confusion(id10m_s.true_idioms, id10m_s.predicted_idioms)
        _update_metrics(results["id10m_metrics"], id10m_ct)

        variant_results = []
        hard_cts = []
        for v in variants:
            vct = analyze_confusion(v.true_idioms, v.predicted_idioms)
            _update_metrics(results["hard_idioms_metrics"], vct)
            hard_cts.append(vct)
            variant_results.append({
                "variant_number": v.variant_number,
                "variant_sentence": v.variant_sentence,
                "predicted_idioms": v.predicted_idioms,
                "confusion_type": vct,
            })

        effect = analyze_context_effect(id10m_ct, hard_cts)
        confused_count = sum(1 for vr in variant_results
                             if vr["confusion_type"] not in ("correct_detection", "correct_rejection"))
        total_v = len(variants)

        comp = {
            "normalized_sentence": norm,
            "true_idioms": id10m_s.true_idioms,
            "id10m": {
                "sentence": id10m_s.sentence,
                "predicted_idioms": id10m_s.predicted_idioms,
                "confusion_type": id10m_ct,
            },
            "hard_idioms_variants": variant_results,
            "context_effect": effect,
            "variant_confusion_stats": {
                "total": total_v,
                "confused": confused_count,
                "confusion_rate": confused_count / total_v if total_v else 0.0,
            },
        }
        results["sentence_comparisons"].append(comp)
        results["context_effects"][effect].append(comp)

        vps = [vr["predicted_idioms"] for vr in variant_results]
        results["vcs_scores"].append(_vcs(vps))
        results["psi_scores"].append(_psi(vps))

    # Advanced metrics
    id10m_m = results["id10m_metrics"]
    hard_m  = results["hard_idioms_metrics"]
    comps   = results["sentence_comparisons"]

    adv = AdvancedMetrics()
    adv.context_degradation_index = (
        (id10m_m.accuracy - hard_m.accuracy) / id10m_m.accuracy
        if id10m_m.accuracy else 0.0
    )
    adv.variant_consistency_score = float(np.mean(results["vcs_scores"])) if results["vcs_scores"] else 0.0

    # CCR
    ccr_num = ccr_den = 0
    for c in comps:
        id10m_ok = c["id10m"]["confusion_type"] in ("correct_detection", "correct_rejection")
        for vr in c["hard_idioms_variants"]:
            ccr_den += 1
            if id10m_ok and vr["confusion_type"] not in ("correct_detection", "correct_rejection"):
                ccr_num += 1
    adv.context_confusion_rate = ccr_num / ccr_den if ccr_den else 0.0

    # LIFR
    lifr_flip = lifr_lit = 0
    for c in comps:
        if c["id10m"]["confusion_type"] == "correct_rejection":
            lifr_lit += 1
            if any(vr["confusion_type"] == "false_positive" for vr in c["hard_idioms_variants"]):
                lifr_flip += 1
    adv.literal_to_idiom_flip_rate = lifr_flip / lifr_lit if lifr_lit else 0.0

    # EAF
    id10m_err = 1 - id10m_m.accuracy
    hard_err  = 1 - hard_m.accuracy
    adv.error_amplification_factor = (
        hard_err / id10m_err if id10m_err else (float("inf") if hard_err else 1.0)
    )

    # Phrase overlap confusion rate
    poc_num = poc_den = 0
    for c in comps:
        sw = set(c["normalized_sentence"].lower().split())
        for vr in c["hard_idioms_variants"]:
            vs = vr.get("variant_sentence") or ""
            vw = set(vs.lower().split())
            if len(sw & vw) > 2:
                poc_den += 1
                if vr["confusion_type"] == "false_positive":
                    poc_num += 1
    adv.phrase_overlap_confusions = poc_num / poc_den if poc_den else 0.0

    # Confusion resistance score
    crs_scores = []
    for c in comps:
        if c["id10m"]["confusion_type"] in ("correct_detection", "correct_rejection"):
            confused = sum(1 for vr in c["hard_idioms_variants"]
                           if vr["confusion_type"] not in ("correct_detection", "correct_rejection"))
            crs_scores.append(confused)
    adv.confusion_resistance_score = float(np.mean(crs_scores)) if crs_scores else 0.0
    adv.prediction_stability_scores = results["psi_scores"]

    results["advanced_metrics"] = adv

    # Variant-level stats
    all_rates = [c["variant_confusion_stats"]["confusion_rate"] for c in comps]
    results["variant_level_stats"] = {
        "average_confusion_rate": float(np.mean(all_rates)) if all_rates else 0.0,
        "median_confusion_rate": float(np.median(all_rates)) if all_rates else 0.0,
        "highly_vulnerable": sum(1 for r in all_rates if r >= 0.8),
        "moderately_vulnerable": sum(1 for r in all_rates if 0.4 <= r < 0.8),
        "resistant": sum(1 for r in all_rates if r < 0.4),
    }

    # Detailed per-idiom confusion (only for id10m-correct sentences)
    results["detailed_confusion"] = _detailed_confusion(comps)

    return results


def _detailed_confusion(comps: list) -> dict:
    per_idiom = defaultdict(lambda: {"confused": 0, "total": 0})
    lit_confused = lit_total = idio_confused = idio_total = 0
    total_v = total_confused_v = 0
    per_sent = []

    for c in comps:
        id10m_ok = c["id10m"]["confusion_type"] in ("correct_detection", "correct_rejection")
        if not id10m_ok:
            continue
        true_idioms = c.get("true_idioms", [])
        is_literal = len(true_idioms) == 0
        row = {
            "sentence": c["normalized_sentence"],
            "true_idioms": true_idioms,
            "id10m_confusion_type": c["id10m"]["confusion_type"],
            "confused_variants": 0,
            "total_variants": len(c["hard_idioms_variants"]),
            "variant_details": [],
        }
        for vr in c["hard_idioms_variants"]:
            is_confused = vr["confusion_type"] not in ("correct_detection", "correct_rejection")
            row["variant_details"].append({
                "variant_number": vr["variant_number"],
                "variant_sentence": vr.get("variant_sentence"),
                "predicted_idioms": vr["predicted_idioms"],
                "confusion_type": vr["confusion_type"],
                "is_confused": is_confused,
            })
            if is_confused:
                row["confused_variants"] += 1
                total_confused_v += 1
            total_v += 1
            if is_literal:
                lit_total += 1
                if is_confused: lit_confused += 1
            else:
                idio_total += 1
                if is_confused: idio_confused += 1
                for idiom in true_idioms:
                    k = idiom.lower().strip()
                    per_idiom[k]["total"] += 1
                    if is_confused: per_idiom[k]["confused"] += 1
        row["confusion_rate"] = row["confused_variants"] / row["total_variants"] if row["total_variants"] else 0.0
        per_sent.append(row)

    per_idiom_out = {
        idiom: {**stats, "confusion_rate": stats["confused"] / stats["total"] if stats["total"] else 0.0}
        for idiom, stats in per_idiom.items()
    }

    return {
        "per_sentence": per_sent,
        "overall_stats": {
            "total_sentences": len(per_sent),
            "total_variants": total_v,
            "total_confused_variants": total_confused_v,
            "overall_confusion_rate": total_confused_v / total_v if total_v else 0.0,
            "sentences_with_any_confusion": sum(1 for s in per_sent if s["confused_variants"] > 0),
            "sentences_with_all_confused": sum(1 for s in per_sent if s["confused_variants"] == s["total_variants"]),
            "sentences_with_no_confusion": sum(1 for s in per_sent if s["confused_variants"] == 0),
        },
        "literal_idiomatic_breakdown": {
            "literal_confused": lit_confused, "literal_total": lit_total,
            "idiomatic_confused": idio_confused, "idiomatic_total": idio_total,
        },
        "per_idiom_breakdown": per_idiom_out,
    }


####################################################################################################
# Report / JSON output

def write_report(results: dict, path: Path):
    id10m_m = results["id10m_metrics"]
    hard_m  = results["hard_idioms_metrics"]
    adv     = results["advanced_metrics"]
    effects = results["context_effects"]
    comps   = results["sentence_comparisons"]
    n       = len(comps)

    def pct(x): return f"{x/n*100:.1f}%" if n else "N/A"

    lines = [
        "ID10M vs Hard_idioms Comparison Report",
        "=" * 50, "",
        "OVERALL PERFORMANCE METRICS (matched sentences only)",
        "-" * 30,
        f"ID10M Dataset:",
        f"  Total sentences: {id10m_m.total}",
        f"  Accuracy:   {id10m_m.accuracy:.3f}",
        f"  Precision:  {id10m_m.precision:.3f}",
        f"  Recall:     {id10m_m.recall:.3f}",
        f"  Specificity:{id10m_m.specificity:.3f}",
        f"  F1-Score:   {id10m_m.f1_score:.3f}",
        f"  Bal. Acc:   {id10m_m.balanced_accuracy:.3f}",
        f"  MCC:        {id10m_m.mcc:.3f}",
        f"  TP:{id10m_m.correct_detection}  TN:{id10m_m.correct_rejection}  FP:{id10m_m.false_positive}  FN:{id10m_m.false_negative}",
        "",
        f"Hard_idioms Dataset (variants of matched sentences):",
        f"  Total rows:  {hard_m.total}",
        f"  Accuracy:   {hard_m.accuracy:.3f}",
        f"  Precision:  {hard_m.precision:.3f}",
        f"  Recall:     {hard_m.recall:.3f}",
        f"  Specificity:{hard_m.specificity:.3f}",
        f"  F1-Score:   {hard_m.f1_score:.3f}",
        f"  Bal. Acc:   {hard_m.balanced_accuracy:.3f}",
        f"  MCC:        {hard_m.mcc:.3f}",
        f"  TP:{hard_m.correct_detection}  TN:{hard_m.correct_rejection}  FP:{hard_m.false_positive}  FN:{hard_m.false_negative}",
        "",
        "CONTEXT EFFECTS SUMMARY",
        "-" * 25,
        f"Total matched sentences: {n}",
        f"Context helped: {len(effects['helped'])} ({pct(len(effects['helped']))})",
        f"Context hurt:   {len(effects['hurt'])} ({pct(len(effects['hurt']))})",
        f"Mixed:          {len(effects['mixed'])} ({pct(len(effects['mixed']))})",
        f"No change:      {len(effects['no_change'])} ({pct(len(effects['no_change']))})",
        "",
        "ADVANCED CONTEXT-CONFUSION METRICS",
        "-" * 35,
        f"Context Degradation Index (CDI): {adv.context_degradation_index:.3f}  ({adv.context_degradation_index*100:.1f}% relative drop)",
        f"Variant Consistency Score (VCS): {adv.variant_consistency_score:.3f}",
        f"Context Confusion Rate (CCR):    {adv.context_confusion_rate:.3f}  ({adv.context_confusion_rate*100:.1f}% of predictions turned wrong)",
        f"Literal-to-Idiom Flip Rate (LIFR): {adv.literal_to_idiom_flip_rate:.3f}",
        f"Error Amplification Factor (EAF): {adv.error_amplification_factor:.3f}",
        f"Phrase Overlap Confusion Rate:   {adv.phrase_overlap_confusions:.3f}",
        f"Confusion Resistance Score:      {adv.confusion_resistance_score:.3f}",
        f"Avg Prediction Stability Index:  {np.mean(adv.prediction_stability_scores):.3f}" if adv.prediction_stability_scores else "",
        "",
        "VARIANT-LEVEL STATS",
        "-" * 20,
    ]
    vls = results["variant_level_stats"]
    lines += [
        f"Avg confusion rate: {vls['average_confusion_rate']:.3f}",
        f"Median:             {vls['median_confusion_rate']:.3f}",
        f"Highly vulnerable (≥80%): {vls['highly_vulnerable']}",
        f"Moderately vulnerable:    {vls['moderately_vulnerable']}",
        f"Resistant (<40%):         {vls['resistant']}",
        "",
    ]

    # Detailed confusion section
    if "detailed_confusion" in results:
        dc = results["detailed_confusion"]
        ov = dc["overall_stats"]
        lib = dc["literal_idiomatic_breakdown"]
        lines += [
            "DETAILED CONFUSION ANALYSIS (id10m-correct sentences only)",
            "=" * 50,
            f"Sentences analyzed: {ov['total_sentences']}",
            f"Total variants:     {ov['total_variants']}",
            f"Confused variants:  {ov['total_confused_variants']} ({ov['overall_confusion_rate']*100:.1f}%)",
            f"  Sentences with any confusion:  {ov['sentences_with_any_confusion']}",
            f"  Sentences all-confused:        {ov['sentences_with_all_confused']}",
            f"  Sentences never confused:      {ov['sentences_with_no_confusion']}",
            "",
            f"Literal variants:   {lib['literal_total']} total, {lib['literal_confused']} confused "
            f"({lib['literal_confused']/lib['literal_total']*100:.1f}%)" if lib['literal_total'] else "Literal: N/A",
            f"Idiomatic variants: {lib['idiomatic_total']} total, {lib['idiomatic_confused']} confused "
            f"({lib['idiomatic_confused']/lib['idiomatic_total']*100:.1f}%)" if lib['idiomatic_total'] else "Idiomatic: N/A",
            "",
            "PER-IDIOM CONFUSION RATES",
            "-" * 25,
            f"{'Idiom':<40} {'Conf':>5} {'Tot':>5} {'Rate':>7}",
            "-" * 60,
        ]
        for idiom, s in sorted(dc["per_idiom_breakdown"].items(), key=lambda x: -x[1]["confusion_rate"]):
            lines.append(f"{idiom:<40} {s['confused']:>5} {s['total']:>5} {s['confusion_rate']*100:>6.1f}%")
        lines.append("")

        lines.append("PER-SENTENCE CONFUSION DETAILS")
        lines.append("-" * 30)
        for s in dc["per_sentence"]:
            lines.append(f"Sentence: {s['sentence']}")
            lines.append(f"True idioms: {s['true_idioms']}")
            lines.append(f"Confused variants: {s['confused_variants']}/{s['total_variants']} ({s['confusion_rate']*100:.0f}%)")
            for vd in s["variant_details"]:
                status = "CONFUSED" if vd["is_confused"] else "ok"
                ctx = (vd.get("variant_sentence") or "")[:90]
                lines.append(f"  v{vd['variant_number']} [{status}] pred={vd['predicted_idioms']} | {ctx}")
            lines.append("-" * 80)

    path.write_text("\n".join(lines), encoding="utf-8")


def write_metrics_json(results: dict, path: Path):
    adv = results["advanced_metrics"]
    effects = results["context_effects"]
    comps = results["sentence_comparisons"]
    n = len(comps)

    def m2d(m: ConfusionMetrics):
        return {k: getattr(m, k) for k in
                ("total","accuracy","precision","recall","specificity","f1_score",
                 "balanced_accuracy","mcc","correct_detection","correct_rejection",
                 "false_positive","false_negative")}

    export = {
        "experiment_info": {
            "total_matched_sentences": n,
            "comparison_timestamp": datetime.now().isoformat(),
        },
        "id10m_metrics": m2d(results["id10m_metrics"]),
        "hard_idioms_metrics": m2d(results["hard_idioms_metrics"]),
        "advanced_metrics": {
            "context_degradation_index": adv.context_degradation_index,
            "variant_consistency_score": adv.variant_consistency_score,
            "context_confusion_rate": adv.context_confusion_rate,
            "literal_to_idiom_flip_rate": adv.literal_to_idiom_flip_rate,
            "error_amplification_factor": adv.error_amplification_factor,
            "phrase_overlap_confusions": adv.phrase_overlap_confusions,
            "confusion_resistance_score": adv.confusion_resistance_score,
            "average_prediction_stability": float(np.mean(adv.prediction_stability_scores)) if adv.prediction_stability_scores else 0.0,
            "prediction_stability_scores": adv.prediction_stability_scores,
        },
        "context_effects": {
            "helped": len(effects["helped"]),
            "hurt": len(effects["hurt"]),
            "mixed": len(effects["mixed"]),
            "no_change": len(effects["no_change"]),
            "pct_helped": len(effects["helped"]) / n * 100 if n else 0,
            "pct_hurt": len(effects["hurt"]) / n * 100 if n else 0,
            "pct_mixed": len(effects["mixed"]) / n * 100 if n else 0,
            "pct_no_change": len(effects["no_change"]) / n * 100 if n else 0,
        },
        "variant_level_stats": results["variant_level_stats"],
        "detailed_confusion": results.get("detailed_confusion", {}),
        "sentence_level_analysis": [
            {
                "normalized_sentence": c["normalized_sentence"],
                "true_idioms": c["true_idioms"],
                "id10m": c["id10m"],
                "hard_idioms_variants": c["hard_idioms_variants"],
                "context_effect": c["context_effect"],
                "variant_confusion_stats": c["variant_confusion_stats"],
            }
            for c in comps
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    return export


####################################################################################################
# model_comparison_table.csv builder

SUMMARY_FIELDS = [
    "model", "prompt_type", "seed", "shots", "sc_runs", "temperature", "language",
    "matched_sentences",
    # id10m
    "id10m_accuracy", "id10m_precision", "id10m_recall", "id10m_f1", "id10m_mcc",
    "id10m_tp", "id10m_tn", "id10m_fp", "id10m_fn",
    # hard_idioms
    "hard_accuracy", "hard_precision", "hard_recall", "hard_f1", "hard_mcc",
    "hard_tp", "hard_tn", "hard_fp", "hard_fn",
    # advanced
    "context_degradation_index", "variant_consistency_score", "context_confusion_rate",
    "literal_to_idiom_flip_rate", "error_amplification_factor",
    "phrase_overlap_confusion_rate", "confusion_resistance_score",
    "avg_confusion_rate", "pct_highly_vulnerable",
    # context effects
    "pct_helped", "pct_hurt", "pct_mixed", "pct_no_change",
    # detailed
    "total_confused_variants", "overall_confusion_rate",
    "literal_confusion_rate", "idiomatic_confusion_rate",
]


def build_summary_row(model, prompt_type, seed, shots, sc_runs, temperature, lang, results):
    id10m_m = results["id10m_metrics"]
    hard_m  = results["hard_idioms_metrics"]
    adv     = results["advanced_metrics"]
    effects = results["context_effects"]
    vls     = results["variant_level_stats"]
    dc      = results.get("detailed_confusion", {})
    ov      = dc.get("overall_stats", {})
    lib     = dc.get("literal_idiomatic_breakdown", {})
    n       = len(results["sentence_comparisons"])

    lit_total  = lib.get("literal_total", 0)
    idio_total = lib.get("idiomatic_total", 0)
    total_v    = ov.get("total_variants", 0)

    return {
        "model": model, "prompt_type": prompt_type, "seed": seed,
        "shots": shots, "sc_runs": sc_runs, "temperature": temperature, "language": lang,
        "matched_sentences": n,
        "id10m_accuracy":  id10m_m.accuracy,
        "id10m_precision": id10m_m.precision,
        "id10m_recall":    id10m_m.recall,
        "id10m_f1":        id10m_m.f1_score,
        "id10m_mcc":       id10m_m.mcc,
        "id10m_tp": id10m_m.correct_detection, "id10m_tn": id10m_m.correct_rejection,
        "id10m_fp": id10m_m.false_positive,    "id10m_fn": id10m_m.false_negative,
        "hard_accuracy":  hard_m.accuracy,
        "hard_precision": hard_m.precision,
        "hard_recall":    hard_m.recall,
        "hard_f1":        hard_m.f1_score,
        "hard_mcc":       hard_m.mcc,
        "hard_tp": hard_m.correct_detection, "hard_tn": hard_m.correct_rejection,
        "hard_fp": hard_m.false_positive,    "hard_fn": hard_m.false_negative,
        "context_degradation_index":   adv.context_degradation_index,
        "variant_consistency_score":   adv.variant_consistency_score,
        "context_confusion_rate":      adv.context_confusion_rate,
        "literal_to_idiom_flip_rate":  adv.literal_to_idiom_flip_rate,
        "error_amplification_factor":  adv.error_amplification_factor,
        "phrase_overlap_confusion_rate": adv.phrase_overlap_confusions,
        "confusion_resistance_score":  adv.confusion_resistance_score,
        "avg_confusion_rate":          vls["average_confusion_rate"],
        "pct_highly_vulnerable":       vls["highly_vulnerable"] / n * 100 if n else 0,
        "pct_helped":   len(effects["helped"])   / n * 100 if n else 0,
        "pct_hurt":     len(effects["hurt"])     / n * 100 if n else 0,
        "pct_mixed":    len(effects["mixed"])    / n * 100 if n else 0,
        "pct_no_change": len(effects["no_change"]) / n * 100 if n else 0,
        "total_confused_variants": ov.get("total_confused_variants", 0),
        "overall_confusion_rate":  ov.get("overall_confusion_rate", 0),
        "literal_confusion_rate":  lib.get("literal_confused", 0) / lit_total if lit_total else 0,
        "idiomatic_confusion_rate": lib.get("idiomatic_confused", 0) / idio_total if idio_total else 0,
    }


####################################################################################################
# Run discovery

def _read_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    import yaml
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def discover_runs(lang: str, filter_str: Optional[str] = None):
    """
    Walk results/hard_idioms/{lang}/updated/ and find matching id10m runs.
    Yields (model, prompt_type, seed, hard_responses_path, id10m_responses_path, cfg).
    """
    hard_base = HARD_DIR / lang / "updated"
    id10m_base = ID10M_DIR / lang

    for model_dir in sorted(hard_base.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if filter_str and filter_str.lower() not in model.lower():
            continue

        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            prompt_type = prompt_dir.name   # zero_shot or few_shot

            for seed_dir in sorted(prompt_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                seed = int(seed_dir.name.split("_")[1])

                hard_resp = seed_dir / "responses.json"
                id10m_resp = id10m_base / model / prompt_type / f"seed_{seed}" / "responses.json"

                if not hard_resp.exists():
                    continue
                if not id10m_resp.exists():
                    logger.debug(f"No id10m match for {model}/{prompt_type}/seed_{seed} — skipping")
                    continue

                cfg = _read_config(seed_dir)
                yield model, prompt_type, seed, hard_resp, id10m_resp, cfg


####################################################################################################
# Main

def run_all(lang: str, filter_str: Optional[str], dry_run: bool):
    out_base = COMPARISONS_DIR / lang
    out_base.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    runs = list(discover_runs(lang, filter_str))
    logger.info(f"[{lang}] Found {len(runs)} matching run pairs")

    for model, prompt_type, seed, hard_resp, id10m_resp, cfg in runs:
        label = f"{model}/{prompt_type}/seed_{seed}"
        if dry_run:
            logger.info(f"  DRY RUN: {label}")
            continue

        out_dir = out_base / model / prompt_type / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Comparing {label} ...")
        try:
            id10m_sents = load_id10m(id10m_resp)
            hard_sents  = load_hard_idioms(hard_resp)
            matches     = find_matches(id10m_sents, hard_sents)

            if not matches:
                logger.warning(f"    No matched sentences — skipping")
                continue

            logger.info(f"    Matched sentences: {len(matches)}")
            results = compare(matches)

            write_report(results, out_dir / "comparison_report.txt")
            write_metrics_json(results, out_dir / "metrics.json")

            shots      = cfg.get("shots", 0)
            sc_runs    = cfg.get("sc_runs", 1)
            temperature = cfg.get("temperature", 0.3)
            row = build_summary_row(model, prompt_type, seed, shots, sc_runs, temperature, lang, results)
            summary_rows.append(row)

            id10m_m = results["id10m_metrics"]
            hard_m  = results["hard_idioms_metrics"]
            adv     = results["advanced_metrics"]
            logger.info(
                f"    id10m F1={id10m_m.f1_score:.3f}  hard F1={hard_m.f1_score:.3f}  "
                f"CDI={adv.context_degradation_index:.3f}  CCR={adv.context_confusion_rate:.3f}"
            )

        except Exception as e:
            logger.error(f"    FAILED: {e}", exc_info=True)

    if not dry_run and summary_rows:
        table_path = out_base / "model_comparison_table.csv"
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(summary_rows)
        logger.info(f"[{lang}] Wrote model_comparison_table.csv ({len(summary_rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["english", "german", "both"], default="both")
    parser.add_argument("--filter", default=None, help="Only run configs where model name contains this string")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "both" else [args.lang]
    for lang in langs:
        run_all(lang, args.filter, args.dry_run)


if __name__ == "__main__":
    main()
