# Hard Idioms Data — English

## Final Dataset

### `hard_idioms_english_FINAL.csv` ✅
**The complete, final English hard idioms dataset. Use this for experiments.**

- 187 unique sentences × 3 variants = **561 rows**
- Columns: `sentence, PIE, true_idioms, is_figurative, variant_number, variant_sentence, tokens, tags, tag_ids, version`
- `version=v1`: 161 sentences — variants generated from the original hard idioms JSON via Gemini
- `version=v2`: 26 sentences — variants manually reviewed and cleaned (from `NEW_english_variants.csv`)
- PIE and `is_figurative` sourced from `id10m_english.csv`

---

## Intermediate / Source Files

### `hard_idioms_english_v1.csv`
First-pass dataset built directly from `ORIGINAL_hard_idioms_data_english.json`.
- 178 sentences × 3 variants = 534 rows, all `version=v1`
- Applied fixes: `hand→head` typo, `bread und→and butter`, removed 8 invalid sentences
- Superseded by `hard_idioms_english_FINAL.csv`

### `NEW_english_variants.csv`
Manually reviewed and cleaned variant sentences (v2 quality).
- 24 sentences × 3 variants = 72 rows
- Contains a `Valid` column (V / X / EXTRA) from the review process
- These rows are incorporated into `hard_idioms_english_FINAL.csv` as `version=v2`

### `ORIGINAL_hard_idioms_data_english.json`
Raw generation output from Gemini (original hard idioms variants before cleaning).
- 185 unique sentences, up to 5 variants each (882 rows total)
- Source for `hard_idioms_english_v1.csv`
