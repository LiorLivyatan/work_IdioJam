#!/usr/bin/env python3
"""
Copy encoder `new_results.json` predictions from Alon's re-run directory
into this repo, split into id10m and hard_idioms subsets.

Source:
    /Users/liorlivyatan/Desktop/Livyatan/Thesis/hard_idioms/encoders/
        {lang}/{model}/{seed}/results/new_results.json

Destination:
    encoders_experiment/{lang}/{model}/seed_{n}/
        new_results_id10m.json        (keys match id10m_{lang}_FINAL.json)
        new_results_hard_idioms.json  (keys match hard_idioms_{lang}_FINAL.json variants)

Invalid sentences (not present in FINAL.json) are excluded automatically.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_BASE = Path("/Users/liorlivyatan/Desktop/Livyatan/Thesis/hard_idioms/encoders")
DEST_BASE = ROOT / "encoders_experiment"

MODELS = {
    "english": [
        "google_bert_bert_base_uncased",
        "google_bert_bert_base_multilingual_cased",
    ],
    "german": [
        "google_bert_bert_base_german_cased",
        "google_bert_bert_base_multilingual_cased",
    ],
}

SEEDS = [5, 7, 42, 123, 1773]

ID10M_FINAL = {
    "english": ROOT / "data" / "raw_id10m_data" / "english" / "id10m_english_FINAL.json",
    "german":  ROOT / "data" / "raw_id10m_data" / "german"  / "id10m_german_FINAL.json",
}

HARD_FINAL = {
    "english": ROOT / "data" / "hard_idioms_data" / "english" / "hard_idioms_english_FINAL.json",
    "german":  ROOT / "data" / "hard_idioms_data" / "german"  / "hard_idioms_german_FINAL.json",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def split_for_language(lang, id10m_sents, hard_variants):
    print(f"\n{'='*60}\n{lang}: id10m={len(id10m_sents)}, hard_variants={len(hard_variants)}\n{'='*60}")

    for model in MODELS[lang]:
        for seed in SEEDS:
            src = SOURCE_BASE / lang / model / str(seed) / "results" / "new_results.json"
            if not src.exists():
                print(f"  MISS: {src}")
                continue

            new_results = load_json(src)

            id10m_preds = {s: new_results[s] for s in id10m_sents if s in new_results}
            hard_preds  = {v: new_results[v] for v in hard_variants if v in new_results}

            id10m_missing = len(id10m_sents) - len(id10m_preds)
            hard_missing  = len(hard_variants) - len(hard_preds)

            dest_dir = DEST_BASE / lang / model / f"seed_{seed}"
            dest_dir.mkdir(parents=True, exist_ok=True)

            write_json(dest_dir / "new_results_id10m.json", id10m_preds)
            write_json(dest_dir / "new_results_hard_idioms.json", hard_preds)

            tag = "OK" if id10m_missing == 0 and hard_missing == 0 else f"MISSING id10m={id10m_missing} hard={hard_missing}"
            print(f"  {model}/seed_{seed:4d}: id10m={len(id10m_preds)}, hard={len(hard_preds)}  [{tag}]")


def main():
    for lang in MODELS:
        id10m_data = load_json(ID10M_FINAL[lang])
        hard_data  = load_json(HARD_FINAL[lang])
        id10m_sents    = set(e["sentence"] for e in id10m_data)
        hard_variants  = set(e["variant_sentence"] for e in hard_data)
        split_for_language(lang, id10m_sents, hard_variants)

    print("\nDone.")


if __name__ == "__main__":
    main()
