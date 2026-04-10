# German Hard Idioms FINAL — Next Steps

## Current Status

English FINAL is complete (561 rows, 30 experiments done).
German FINAL is complete (411 rows). Experiments NOT yet run.

---

## ✅ Step 1: Review & add the 14 new generated variants (DONE)

- Generated variants for 14 sentences in `german_gemini_2_5_pro_all_14s_5v_20260408_185115.csv`
- Reviewed and added approved (Valid=V) rows to `NEW_german_variants.csv`
- NEW_german_variants.csv now has 18 sentences × 3 variants = 54 rows (v2)

---

## ✅ Step 2: Build `hard_idioms_german_FINAL.csv` + `.json` (DONE)

Built at:
- `data/hard_idioms_data/german/hard_idioms_german_FINAL.csv`
- `data/hard_idioms_data/german/hard_idioms_german_FINAL.json`

Final stats:
- **411 rows** (137 sentences × 3 variants), all sentences covered
- v2 (NEW/generated): 54 rows (18 sentences)
- v1 (ORIGINAL JSON): 357 rows (119 sentences)
- id10m reduced from 146→137 sentences (9 removed — no good variants)
- ORIGINAL sentences with non-sequential variant numbers: took first 3 available, renumbered 1,2,3

---

## ~~Step 3: Create `final_vs_old_comparison.csv`~~ — NOT NEEDED

No old German archive runs exist at `results/hard_idioms/german/archive/`.
All 411 German rows are new — no merge logic required.

---

## ~~Step 4: Create `hard_idioms_german_FINAL_subset.json`~~ — NOT NEEDED

No old responses to compare against. The batch runner uses the full FINAL JSON directly.

---

## Step 5: Run German experiments

Use `experiments/run_hard_batch_german.py` (already created).

```bash
# Dry run — see what would run
python experiments/run_hard_batch_german.py --dry_run

# Run all experiments
python experiments/run_hard_batch_german.py

# Run specific model only
python experiments/run_hard_batch_german.py --filter gpt-4o-mini
```

Results land in `results/hard_idioms/german/updated/`.
Summary CSV auto-updated after each run at `results/hard_idioms/german/updated/results_summary.csv`.

Experiment matrix (same as English):
- gemini-2.5-flash-lite: zero_shot + few_shot, seeds 42/43/44
- gpt-4o: zero_shot + few_shot, seed 42
- gpt-4o-mini: zero_shot + few_shot, seeds 42/43/44
- llama-4-scout: zero_shot + few_shot, seeds 42/43/44
- qwen-2.5-72b-instruct: zero_shot + few_shot, seeds 42/43/44
- gemini-2.5-pro: zero_shot + few_shot, seed 42
- o3-mini: zero_shot, seed 42
- deepseek-ai/DeepSeek-R1: zero_shot, seed 42
