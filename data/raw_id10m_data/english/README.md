# ID10M Data — English

## Final Dataset

### `id10m_english.csv` ✅
**The complete, final English ID10M dataset. Use this for experiments.**

- 194 sentences
- Columns: `sentence, PIE, true_idioms, is_figurative, tokens, tags, tag_ids, is_valid`
- `is_valid=True` (180 sentences): sentences suitable for experiments
- `is_valid=False` (14 sentences): excluded sentences (literal noise, duplicates, or problematic phrasing)
- PIE values use corrected spelling (`bread and butter`, `bury their head in the sand`)

---

## Intermediate / Source Files

### `id10m_english_with_pie.csv`
Manual annotation file used to build the final dataset.
- 194 sentences with columns: `sentence, PIEs, true_idioms, Valid, Comments`
- `Valid` and `Comments` reflect human review decisions
- Source for PIE values in `id10m_english.csv`

### `parsed_test_english_full.csv`
Parsed tokenisation output for all 200 original test sentences.
- 200 rows with columns: `sentence, tokens, tags, tag_ids, true_idioms`
- Source for token/tag columns in `id10m_english.csv`

### `test_english.tsv`
Original raw ID10M test split in TSV format (200 sentences).

### `parsed_test_english.txt`
Intermediate parsed output (text format) produced during data processing.

### `used_sentences.txt` / `unused_sentences.txt`
Bookkeeping files tracking which of the 200 original sentences were included
in or excluded from `id10m_english.csv`.
