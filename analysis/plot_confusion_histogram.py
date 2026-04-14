"""
Plot histogram of the number of models that experienced a negative context drift
per hard variant (Figure 3 in the paper).

"Negative drift" = model was confused on the hard variant (FP or FN) but was
correct on the corresponding plain id10m sentence.  Encoder models (haiku) are
excluded.

For each language we:
  1. Find every model that has BOTH hard_idioms and id10m results for the
     chosen prompt_type / seed.
  2. Match hard variants to id10m sentences by normalized sentence text.
  3. Per variant: count how many models show negative drift.
  4. Plot a histogram of those counts.

Usage:
    python analysis/plot_confusion_histogram.py
    python analysis/plot_confusion_histogram.py --prompt_type zero_shot --seed 42
    python analysis/plot_confusion_histogram.py --prompt_type few_shot  --seed 42
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent

HARD_DIR  = REPO_ROOT / "results" / "hard_idioms"
ID10M_DIR = REPO_ROOT / "results" / "id10m"
PLOTS_DIR = REPO_ROOT / "results" / "plots"

# Models to exclude (encoder-only baselines)
EXCLUDE_MODELS = {"claude-3-haiku-20240307"}

# dir-name aliases: hard_idioms uses different casing than id10m in some models
HARD_TO_ID10M_ALIASES = {
    "DeepSeek-R1": "deepseek-r1",
}

LABEL_FONT = 14


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def normalize(s: str) -> str:
    s = re.sub(r"\s*―\s*.*$", "", s)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([.!?,:;])", r"\1", s)
    return s


def extract_predicted(item: dict) -> list:
    """Return list of predicted idiom strings from a responses.json entry."""
    def _parse(v):
        if isinstance(v, list):  return [str(x) for x in v if x is not None]
        if isinstance(v, str):   return [v] if v else []
        return []
    try:
        resp = item.get("responses", [{}])[0]
        parsed = resp.get("parsed")
        if isinstance(parsed, dict):
            return _parse(parsed.get("idioms", []))
        if isinstance(parsed, list):
            return _parse(parsed)
        if "idioms" in resp:
            return _parse(resp["idioms"])
    except Exception:
        pass
    return []


def coerce_list(val) -> list:
    if isinstance(val, list): return val
    if not isinstance(val, str) or not val: return []
    s = val.strip()
    if not (s.startswith("[") and s.endswith("]")): return [s]
    tokens = re.findall(r"['\"]([^'\"]*)['\"]", s)
    return tokens if tokens else []


def is_confused(true_idioms: list, pred_idioms: list) -> bool:
    """Return True if the prediction is wrong (FP or FN)."""
    has_true = bool(true_idioms)
    has_pred = bool(pred_idioms)
    if not has_true and not has_pred: return False   # correct rejection
    if not has_true and has_pred:     return True    # FP
    if has_true and not has_pred:     return True    # FN
    t = [t.lower().strip() for t in true_idioms]
    p = [x.lower().strip() for x in pred_idioms]
    for ti in t:
        if not any(ti in pi or pi in ti for pi in p):
            return True   # FN on this idiom
    return False


# ── per-language analysis ─────────────────────────────────────────────────────

def compute_variant_drifts(lang: str, prompt_type: str, seed: int):
    """
    Returns a dict  variant_sentence -> int  (number of models with negative drift).
    """
    hard_lang_dir  = HARD_DIR  / lang / "updated"
    id10m_lang_dir = ID10M_DIR / lang / "updated"

    prompt_dir   = "few_shot" if "few_shot" in prompt_type else "zero_shot"
    seed_dir     = f"seed_{seed}"

    # Collect model dirs that have both datasets
    model_hard_dirs = {d.name: d for d in hard_lang_dir.iterdir() if d.is_dir()
                       and d.name != "results_summary.csv"}

    drift_counts: dict[str, int] = defaultdict(int)   # variant_sentence -> count
    models_used  = []

    for hard_model_name, hard_base in sorted(model_hard_dirs.items()):
        if hard_model_name in EXCLUDE_MODELS:
            continue

        hard_resp_path = hard_base / prompt_dir / seed_dir / "responses.json"
        if not hard_resp_path.exists():
            continue

        # Find matching id10m dir (handle casing aliases)
        id10m_model_name = HARD_TO_ID10M_ALIASES.get(hard_model_name, hard_model_name)
        id10m_resp_path  = id10m_lang_dir / id10m_model_name / prompt_dir / seed_dir / "responses.json"
        if not id10m_resp_path.exists():
            continue

        models_used.append(hard_model_name)

        # Build id10m lookup:  normalized_sentence -> confused?
        id10m_data   = load_json(id10m_resp_path)
        id10m_lookup = {}
        for item in id10m_data:
            norm  = normalize(item.get("sentence", ""))
            true  = coerce_list(item.get("true_idioms", []))
            pred  = extract_predicted(item)
            id10m_lookup[norm] = is_confused(true, pred)

        # Iterate hard variants
        hard_data = load_json(hard_resp_path)
        for item in hard_data:
            variant_sent = item.get("variant_sentence", "").strip()
            orig_norm    = normalize(item.get("sentence", ""))
            true         = coerce_list(item.get("true_idioms", []))
            pred         = extract_predicted(item)

            hard_confused = is_confused(true, pred)
            id10m_confused = id10m_lookup.get(orig_norm, False)

            # Negative drift: confused on hard variant but correct on plain sentence
            if hard_confused and not id10m_confused:
                drift_counts[variant_sent] += 1

    print(f"  [{lang}] Models used ({len(models_used)}): {models_used}")
    return drift_counts


def plot_histogram(drift_counts: dict, lang: str, prompt_type: str, seed: int, out_dir: Path):
    counts = list(drift_counts.values())
    counts = [c for c in counts if c > 0]

    if not counts:
        print(f"  [{lang}] No variants with drift — skipping plot")
        return

    max_val = max(counts)
    bins    = range(1, max_val + 2)
    freq    = {k: counts.count(k) for k in range(1, max_val + 1)}

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(list(freq.keys()), list(freq.values()), width=0.8, align="center",
           color="#1f77b4", edgecolor="white")

    for x, y in freq.items():
        if y > 0:
            ax.text(x, y + 1.5, str(y), ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Number of Models Confused", fontsize=LABEL_FONT)
    ax.set_ylabel("Number of Hard Variants",   fontsize=LABEL_FONT)
    ax.set_xticks(range(0, max_val + 1))
    ax.set_xlim(-0.5, max_val + 0.5)
    ax.set_ylim(0, max(freq.values()) * 1.15 + 5)

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{lang}_confusion_histogram_{prompt_type}_seed{seed}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  [{lang}] Saved → {out_dir / fname}")
    print(f"         Variants with drift: {len(counts)}  |  max drift: {max_val}")


# ── side-by-side figure ───────────────────────────────────────────────────────

def plot_side_by_side(drift_en: dict, drift_de: dict, prompt_type: str, seed: int, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, drift_counts, title in [
        (axes[0], drift_en, "(a) English"),
        (axes[1], drift_de, "(b) German"),
    ]:
        counts = [c for c in drift_counts.values() if c > 0]
        if not counts:
            ax.set_title(title, fontsize=LABEL_FONT)
            continue

        max_val = max(counts)
        freq    = {k: counts.count(k) for k in range(1, max_val + 1)}

        ax.bar(list(freq.keys()), list(freq.values()), width=0.8, align="center",
               color="#1f77b4", edgecolor="white")

        for x, y in freq.items():
            if y > 0:
                ax.text(x, y + 1.5, str(y), ha="center", fontsize=10, fontweight="bold")

        ax.set_xlabel("Number of Models Confused", fontsize=LABEL_FONT)
        ax.set_ylabel("Number of Hard Variants",   fontsize=LABEL_FONT)
        ax.set_xticks(range(0, max_val + 1))
        ax.set_xlim(-0.5, max_val + 0.5)
        ax.set_ylim(0, max(freq.values()) * 1.15 + 5)
        ax.set_title(title, fontsize=LABEL_FONT)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"confusion_histogram_sidebyside_{prompt_type}_seed{seed}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"\nSide-by-side saved → {out_dir / fname}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", default="few_shot",
                        choices=["zero_shot", "few_shot"],
                        help="Prompt type to use (default: few_shot)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Computing confusion drifts — prompt={args.prompt_type}  seed={args.seed}")

    drift_en = compute_variant_drifts("english", args.prompt_type, args.seed)
    drift_de = compute_variant_drifts("german",  args.prompt_type, args.seed)

    plot_histogram(drift_en, "english", args.prompt_type, args.seed, PLOTS_DIR)
    plot_histogram(drift_de, "german",  args.prompt_type, args.seed, PLOTS_DIR)
    plot_side_by_side(drift_en, drift_de, args.prompt_type, args.seed, PLOTS_DIR)


if __name__ == "__main__":
    main()
