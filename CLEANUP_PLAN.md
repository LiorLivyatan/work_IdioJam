# Cleanup Plan: Remove 9 Invalid English Sentences

## Background

9 sentences in `data/raw_id10m_data/english/id10m_english.csv` are marked `is_valid=False`.
These were incorrectly included in all downstream experiments. This plan removes them
from every dataset, response file, and derived output — **English only** (German is unaffected).

**Scope summary:**
- 9 invalid sentences in id10m CSV (187 → 178 rows)
- 27 hard_idioms variants to remove (3 per sentence × 9) (561 → 534)
- 7 items in subset105 to remove (105 → 98)
- 32 hard_idioms response files to filter + re-evaluate
- 38 id10m response files to filter + re-evaluate
- All English comparison CSVs to regenerate (4 tables + per-model reports)
- Histogram plots to regenerate

**Invalid sentences:**
1. The ship broke the ice all the way.
2. I thought you rang the bell.
3. Would you like a piece of cake.
4. Is soaking in hot water good for us?
5. What can I bring you? Maybe bread and butter?
6. The submarine is getting into deep water.
7. The fairy tale tells about a caste in the sky where a king and a queen live.
8. They asked me if we can go dutch.
9. The old computer just doesn't hold a candle to the latest models.

---

## Phase 1 — Data Files

- [x] **1a.** Remove 9 invalid rows from `data/raw_id10m_data/english/id10m_english.csv`
  - Expected: 187 → 178 rows
- [x] **1b.** Remove 27 matching variants from `data/hard_idioms_data/english/hard_idioms_english_FINAL.csv`
  - Expected: 561 → 534 rows
- [x] **1c.** Remove 27 matching items from `data/hard_idioms_data/english/hard_idioms_english_FINAL.json`
  - Expected: 561 → 534 items
- [x] **1d.** Remove 7 matching items from `data/hard_idioms_data/english/hard_idioms_english_FINAL_subset105.json`
  - Expected: 105 → 98 items
- [x] **1e.** Verify counts after each removal (assert exact numbers)

---

## Phase 2 — Filter Hard Idioms Responses

32 response files under `results/hard_idioms/english/updated/` (all 561 → 534 rows).

- [x] **2a.** Write a filter script `scripts/filter_responses_english.py` that:
  - Accepts a `responses.json` path
  - Removes rows where `sentence` matches any of the 9 invalid sentences
  - Saves the filtered file back in-place (after backing up the original)
- [x] **2b.** Run the filter script on all 32 hard_idioms response files
- [x] **2c.** Verify each filtered file has exactly 534 rows

---

## Phase 3 — Recompute Hard Idioms Metrics

For each of the 32 filtered runs, recompute `metrics.json`, `results.tsv`, `conf_matrices_reports.txt` using the updated FINAL.json (from Phase 1c) as ground truth.

- [x] **3a.** Verify `run_exp_hard.py --responses_dir` works with updated data file
- [x] **3b.** Run re-evaluation for all 32 hard_idioms runs (via `scripts/reeval_hard_idioms.py`)
  - Models: DeepSeek-R1, claude-sonnet-4-20250514, gemini-2.5-flash-lite, gemini-2.5-pro, gpt-4o, gpt-4o-mini, llama-4-scout, o3-mini, qwen-2.5-72b-instruct
- [x] **3c.** Verify metrics files updated (check a few spot samples)

---

## Phase 4 — Filter id10m Responses

38 response files under `results/id10m/english/updated/` (all 187 → 178 rows).

- [x] **4a.** Run the filter script on all 38 id10m response files
- [x] **4b.** Verify each filtered file has exactly 178 rows

---

## Phase 5 — Recompute id10m Metrics

- [x] **5a.** Verify `run_exp.py --responses_dir` (or equivalent) works with updated id10m CSV
- [x] **5b.** Run re-evaluation for all 38 id10m runs (via `scripts/reeval_id10m.py`)
  - Models: claude-3-haiku-20240307, claude-sonnet-4-20250514, deepseek-r1, gemini-2.5-flash-lite, gemini-2.5-pro, gpt-4o, gpt-4o-mini, llama-4-scout, o3-mini, qwen-2.5-72b-instruct
- [x] **5c.** Verify metrics files updated

---

## Phase 6 — Regenerate Comparison Tables (English)

- [x] **6a.** Rerun `analysis/compare_id10m_vs_hard_idioms.py` for English (all models)
  - Outputs: `results/comparisons/english/model_comparison_table.csv`,
    `sentence_confusion_table.csv`, per-model `comparison_report.txt` + `metrics.json`
- [x] **6b.** Rerun `analysis/generate_legacy_comparison_table.py` for English
  - Output: `results/comparisons/english/model_comparison_table_legacy.csv`
- [x] **6c.** Rerun `analysis/sentence_confusion_analysis.py` for English (if applicable)

---

## Phase 7 — Regenerate Histogram Plots

- [x] **7a.** Rerun `analysis/plot_confusion_histogram.py` (zero_shot + few_shot)
  - Regenerates `variant_confusion_summary_{zero_shot,few_shot}_seed42.csv` for English
  - Regenerates all 6 PNG plots (English + German individual + side-by-side)
  - Note: German plots will be unchanged but are regenerated for consistency

---

## Phase 8 — Commit

- [x] **8a.** Review all changed files
- [x] **8b.** Commit with message describing the cleanup

---

## Notes

- **German data is NOT affected** — only English.
- **Backup before filtering**: Keep `responses_backup.json` in each run directory (or rely on git).
- **Evaluation script dependency**: Phases 3 and 5 require the data files from Phase 1 to be updated first, since `process_responses` loads test data from the FINAL JSON/CSV.
- **subset105.json**: Used only by `complete_english_few_shot_sc.py` (a one-time patching script). Updating it is for correctness but does not affect any re-evaluation.
