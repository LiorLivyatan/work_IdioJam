# id10m Update Tasks

Goal: Fix id10m results so they cover exactly the same sentences as hard_idioms FINAL,
then run the hard_idioms vs id10m comparison.

---

## English id10m (187 sentences)

Mismatches found:
- 7 sentences removed from dataset after id10m was run → already removed from CSV
- 8 sentences have typos in id10m responses (e.g. "hand"→"head", "that"→"than")
  → need to re-run those 8 sentences for every model/prompt/seed config

### Step 1 — Clean CSV
- [x] Remove 7 dropped sentences from `data/raw_id10m_data/english/id10m_english.csv` (194 → 187)

### Step 2 — Archive existing results
- [x] Move `results/id10m/english/{model dirs}` → `results/id10m/english/archive/`

### Step 3 — Build batch runner
- [x] Created `experiments/run_id10m_batch_english.py`
  - Built `data/raw_id10m_data/english/id10m_english_FINAL.json` (187 rows)
  - Built `data/raw_id10m_data/english/id10m_english_subset8.json` (8 corrected rows)
  - Pipeline tested end-to-end (merge + eval) — F1 improves from 0.8336→0.8579

### Step 4 — Dry run & verify
- [x] `python experiments/run_id10m_batch_english.py --dry_run`
  - 38 experiments, all archive dirs resolve correctly

### Step 5 — Run English batch
- [x] `/opt/miniconda3/bin/python3 experiments/run_id10m_batch_english.py`

### Step 6 — Verify & generate summary
- [x] Verify each updated run has exactly 187 sentences in responses.json
- [x] Verify `results/id10m/english/updated/results_summary.csv` has 38 rows

---

## German id10m (137 sentences)

Mismatches found:
- 63 sentences in id10m responses that are NOT in current CSV (old dataset, to drop)
- 5 sentences in current CSV not in id10m responses (text cleaned/corrected, need re-run):
  1. "Das war eine ganz harmlose Bemerkung. Daraus kann dir niemand einen Strick drehen."
     (response had "(Standard German proper)" appended)
  2. "Immer mit der Ruhe, alter Knabe..." (response had »« quote format)
  3. "Vorbereitung ist die halbe Miete." (response had no period)
  4. "Völlig ausgeschlossen, daß dieser Mann..." (response had »« quote format)
  5. "eine an den Haaren herbeigezogene Behauptung" (response had "― a far-fetched claim" appended)

### Step 7 — Archive existing results
- [x] Move `results/id10m/german/{model dirs}` → `results/id10m/german/archive/`

### Step 8 — Build German batch runner
- [x] Create `experiments/run_id10m_batch_german.py`
  - Infers only on the 5 corrected sentences per config
  - Merges with 132 archive responses (filtered to exact-match sentences only)
  - Recalculates metrics on full 137-sentence set
  - Outputs to `results/id10m/german/updated/{model}/{prompt}/seed_{N}/`

### Step 9 — Dry run & verify
- [x] `python experiments/run_id10m_batch_german.py --dry_run`

### Step 10 — Run German batch
- [x] `python experiments/run_id10m_batch_german.py`

### Step 11 — Verify & generate summary
- [x] Verify each updated run has exactly 137 sentences in responses.json
- [x] Generate `results/id10m/german/updated/results_summary.csv`

---

## Comparisons

### Step 12 — Run English comparison
- [x] `python analysis/compare_id10m_vs_hard_idioms.py --lang english`
- [x] Verify output in `results/comparisons/english/`

### Step 13 — Run German comparison
- [x] `python analysis/compare_id10m_vs_hard_idioms.py --lang german`
- [x] Verify output in `results/comparisons/german/`

### Step 14 — Final check
- [x] Verify `results/comparisons/{lang}/model_comparison_table.csv` has 30 rows (per language)
  - English: 30 rows ✓
  - German: 30 rows ✓
  - 8 runs excluded (claude-3-haiku × 6, claude-sonnet × 2) — no hard_idioms counterpart
