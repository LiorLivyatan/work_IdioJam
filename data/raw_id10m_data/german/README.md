# ID10M Data — German

## Final Dataset

### `id10m_german.csv` ✅
**The complete, final German ID10M dataset. Use this for experiments.**

- 155 sentences
- Columns: `sentence, PIE, true_idioms, is_figurative, tokens, tags, tag_ids, was_fixed`
- 141 figurative, 14 literal sentences
- `was_fixed=True`: sentences where the PIE annotation was corrected during post-processing

---

## Intermediate / Source Files

### `id10m_german_with_pie.csv`
Manual annotation file used to build the final dataset.
- 200 sentences with columns: `sentence, PIEs, true_idioms, Valid, Comments`
- `Valid` and `Comments` reflect human review decisions
- Source for PIE values in `id10m_german.csv`

### `parsed_test_german_full.csv`
Parsed tokenisation output for all 200 original German test sentences.
- 200 rows with columns: `sentence, tokens, tags, tag_ids, true_idioms`
- Source for token/tag columns in `id10m_german.csv`

### `test_german.tsv`
Original raw ID10M German test split in TSV format (200 sentences).

### `parsed_test_german.txt`
Intermediate parsed output (text format) produced during data processing.
