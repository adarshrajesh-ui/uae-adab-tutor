# UAE Adab SLM process record

This folder is the human-readable audit trail for the project. It records what
was attempted, what changed, which artifacts are authoritative, how each model
stacked up, and which limitations remain.

## Naming used in this report

Three releases have been called "v1" or "v2" at different points. They are
not interchangeable:

- **Historical synthetic v1** is the original 600-case, dual-register corpus.
  It contains no source-grounded cases and is not the selected final model.
- **Exact-silver v1 (20% grounded)** is the later 600-case, single-register
  corpus with 120 source-incorporating grounded-silver cases and 480 revised
  synthetic cases. Its Complete-600 adapter is the selected final/demo model.
- **Targeted v2** keeps the same 120 grounded cases and replaces 60 of
  exact-silver v1's 480 revised synthetic cases. It adds no new grounded data.
  Its controlled regression score was worse, so it was not selected.

## Current status on 2026-07-12

- The primary behavior is fixed: a Qwen3-4B academic tutor should preserve UAE
  adab as a teaching method across pressured five-turn lessons.
- Frozen exact-silver v1 (20% grounded) has 600 Luna-written conversations: 120
  source-incorporating grounded-silver cases and 480 source-free revised
  synthetic cases. The 20% figure describes provenance, not human authorship.
- Fresh fidelity review accepted 111 of the 120 grounded rows; eight were
  rejected and one was unreviewed. The full 20% is traceable experimental
  silver, not 20% independently accepted gold.
- The exact-silver v1 Complete-600 and Grounded-120 QLoRA adapters were trained
  independently from the same pinned Qwen base.
- The primary frozen behavior evaluation of those exact-silver v1 adapters is
  complete.
- The exact-silver v1 Complete-600 deployment beat strongly prompted base Qwen
  by 64% versus 18% strict-turn passing. This is the assignment comparison,
  not an adapter-only isolation: the tuned condition also uses its deployment
  token.
- Exact-silver v1 Complete-600 approached, but did not beat, GPT-5.6 Luna with
  the strong prompt when both were scored by the same blind Claude judge.
- The frozen test inventory has 31 five-turn scenarios (10 primary, 20
  source-transfer, 1 authentic held-out). Only the primary 10 have been run.
- Choice #1 is complete as a separate targeted-v2 sibling dataset. Exactly 60
  revised rows were replaced: 37 provisional rows and 23 accepted rows, split
  52 train / 8 validation and balanced as ten rows across six repair
  categories. All 120 grounded rows and all held-out suites remain unchanged.
- All 60 replacements passed both independent reviewers
  (`claude-group/claude-opus-4-8` and
  `gemini-group/gemini-3.1-pro`). Selection among passing variants used a
  stable hash rule, not judge-score cherry-picking.
- Targeted v2 contains 588 accepted rows and 12 disclosed provisional rows;
  all 12 provisional rows are grounded. Exact-silver v1 had 551 accepted and
  49 provisional rows. V2 passed the independent release audit with zero
  errors and zero frozen-eval overlap. Its exact Qwen token maximum is 1,899
  (p95 1,431), with no rows above 2,048.
- Targeted v2 was trained with the same pinned base, seed, exposure order, and
  75-step QLoRA schedule as exact-silver v1. Its controlled primary-regression
  result was worse: 8.98 versus 9.42 mean and 58% versus 66% strict-turn passing.
  V2 added one religious-boundary breach and one late-turn test-answer leak.
  Exact-silver v1 Complete-600 remains the final/demo model.
- Because the 60 v2 repair categories came from exact-silver v1 errors on the
  primary ten scenarios, that suite is now a development/regression set for v2
  even though its text remains frozen. An unbiased v2 generalization claim
  requires the still-unrun 20 source-transfer cases, authentic held-out case,
  or new locked cases.
- The selected adapter and 540/60 dataset are now populated and publicly
  readable on Hugging Face. Their public bytes match the frozen local hashes.
  The GitHub destination is still an empty repository, so the prepared Colab
  reviewer URL will not run until the curated project is pushed there.
- The delivery choice is implemented locally as a bare multi-turn Gradio
  reviewer demo in `deploy/reviewer_demo/app.py` and canonical notebook 03.
  It pins the selected adapter to immutable Hugging Face commit
  `c20d382f32810deaff2f691cdf78c0a3a4d9be59`, verifies both adapter files,
  uses 4-bit Qwen on a T4-or-better Colab, and supplies no prose behavior
  prompt beyond the learned fixed control token.

Files:

- ITERATION_LEDGER.md: single-table chronology of every major data, training,
  evaluation, selection, and release iteration.
- PROJECT_HISTORY.md: chronological narrative, run distinctions, caveats, and
  iteration state.
- COMPARISONS.md: controlled evaluation tables and interpretation.
- ARTIFACT_INDEX.md: authoritative files, hashes, and known limitations.
- TARGETED_V2_REPAIR.md: completed choice-#1 selection, generation, review,
  materialization, and independent-audit record.
- TARGETED_V2_EVAL.md: v2 training, controlled head-to-head result, material
  failures, and the decision to retain exact-silver v1.
- `deploy/reviewer_demo/app.py`: frozen minimal multi-turn reviewer runtime.
- `submission/notebooks/03_demo_uae_adab_tutor.ipynb`: one-click temporary
  Colab/Gradio path; it becomes live when the GitHub repository is populated.

`brainlift.md` is currently a historical/incomplete draft describing the old
dual-register synthetic release and historical synthetic-v1 evaluation. It
must not be treated as the final project account until updated.
`modelBrainlift.md` remains the
untouched formatting example.
