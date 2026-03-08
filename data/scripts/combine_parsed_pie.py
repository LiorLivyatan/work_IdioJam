#!/usr/bin/env python3
"""
Combine parsed_test_english_full.csv with id10m_english_with_pie.csv:
- Adds a PIE column to the parsed data
- Applies sentence fixes carried over from the with_pie edits
- Filters out rows that were removed from with_pie
- Overrides true_idioms with the manually corrected values from with_pie
"""

import ast
import os
import re
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA       = os.path.join(_HERE, "..", "raw_id10m_data", "english")
PARSED_PATH = os.path.join(_DATA, "parsed_test_english_full.csv")
PIE_PATH    = os.path.join(_DATA, "id10m_english_with_pie.csv")
OUTPUT_PATH = os.path.join(_DATA, "id10m_english_combined.csv")

# ── Sentence fixes applied to the with_pie CSV ────────────────────────────────
SENTENCE_FIXES = {
    "The bullet hit thebook in his pocket.":             "The bullet hit the book in his pocket.",
    "Ostriches bury their hand in the sand.":            "Ostriches bury their head in the sand.",
    "Do some animals really bury their hand in the sand?": "Do some animals really bury their head in the sand?",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_list_col(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or str(val).strip() in ("[]", ""):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []

def normalize(text):
    """Collapse multiple spaces into one."""
    return re.sub(r"\s+", " ", str(text)).strip()

def fix_token_row(row):
    """
    Apply token-level fixes for sentences whose wording changed:
      - 'thebook ' → 'the ', 'book '  (one token splits into two; tags/tag_ids duplicated)
      - 'hand '    → 'head '           (in-place replacement, count unchanged)
    Must be called AFTER sentence fix is applied and list columns are parsed.
    """
    sentence = row["sentence"]
    tokens   = list(row["tokens"])
    tags     = list(row["tags"])
    tag_ids  = list(row["tag_ids"])

    if sentence == "The bullet hit the book in his pocket.":
        new_tokens, new_tags, new_tag_ids = [], [], []
        for tok, tag, tid in zip(tokens, tags, tag_ids):
            if tok.strip() == "thebook":
                new_tokens  += ["the ", "book "]
                new_tags    += [tag, tag]
                new_tag_ids += [tid, tid]
            else:
                new_tokens.append(tok)
                new_tags.append(tag)
                new_tag_ids.append(tid)
        row = row.copy()
        row["tokens"], row["tags"], row["tag_ids"] = new_tokens, new_tags, new_tag_ids

    elif sentence in (
        "Ostriches bury their head in the sand.",
        "Do some animals really bury their head in the sand?",
    ):
        row = row.copy()
        row["tokens"] = ["head " if tok.strip() == "hand" else tok for tok in tokens]

    return row

# ── Load parsed CSV ───────────────────────────────────────────────────────────
parsed_df = pd.read_csv(PARSED_PATH)
parsed_df["true_idioms"] = parsed_df["true_idioms"].apply(parse_list_col)
parsed_df["tokens"]      = parsed_df["tokens"].apply(parse_list_col)
parsed_df["tags"]        = parsed_df["tags"].apply(parse_list_col)
parsed_df["tag_ids"]     = parsed_df["tag_ids"].apply(parse_list_col)

# Apply sentence text fixes so the join key matches the with_pie CSV
parsed_df["sentence"] = parsed_df["sentence"].apply(lambda s: SENTENCE_FIXES.get(s, s))

# Apply token-level fixes for the same affected rows
parsed_df = parsed_df.apply(fix_token_row, axis=1)

# ── Load with_pie CSV ─────────────────────────────────────────────────────────
pie_df = pd.read_csv(PIE_PATH)
pie_df["PIEs"]        = pie_df["PIEs"].apply(parse_list_col)
pie_df["true_idioms"] = pie_df["true_idioms"].apply(parse_list_col)

# Build sentence → (PIE string, corrected true_idioms) mappings
pie_map         = {}
true_idioms_map = {}
for _, row in pie_df.iterrows():
    sentence = row["sentence"]
    pies = row["PIEs"]
    pie_map[sentence]         = [normalize(p) for p in pies]
    true_idioms_map[sentence] = [normalize(t) for t in row["true_idioms"]]

# ── Merge ─────────────────────────────────────────────────────────────────────
valid_sentences = set(pie_map.keys())

combined_df = parsed_df[parsed_df["sentence"].isin(valid_sentences)].copy()
combined_df["PIE"]         = combined_df["sentence"].map(pie_map)
combined_df["true_idioms"] = combined_df["sentence"].map(true_idioms_map)

combined_df = combined_df[["sentence", "PIE", "true_idioms", "tokens", "tags", "tag_ids"]]
combined_df = combined_df.reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
combined_df.to_csv(OUTPUT_PATH, index=False)

n_total     = len(combined_df)
n_figurative = combined_df["true_idioms"].apply(lambda x: len(x) > 0).sum()
n_literal    = n_total - n_figurative

print(f"Saved {n_total} rows to {OUTPUT_PATH}")
print(f"  Filtered out: {len(parsed_df) - n_total} rows (removed from with_pie)")
print(f"  Figurative (true_idioms non-empty): {n_figurative}")
print(f"  Literal    (true_idioms empty):     {n_literal}")
