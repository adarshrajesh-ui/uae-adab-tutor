# Targeted v2 repair: choice #1

Status: **completed negative iteration. Dataset repair, dual review,
sibling-release materialization, token audit, independent integrity audit,
training, and controlled regression evaluation are complete. Exact-silver v1
remains the selected final model.**

## Scope and reason

Choice #1 was the data-first iteration: repair known exact-silver v1
Complete-600 failure categories before running another training job. Frozen
exact-silver v1 and all held-out evaluations were left byte-for-byte unchanged.
The sibling release replaces exactly 60 revised-synthetic rows while preserving the
600-row total, all 120 grounded rows, the 540/60 split, and the fixed default
voice.

The repair categories came from observed exact-silver v1 model failures rather
than copied held-out answers:

1. calibrated directness versus excessive scaffolding;
2. transparent full verification after tutor error;
3. peer-harm repair and usable apology language;
4. respectful factual correction of parent or teacher claims;
5. factual precision using new subject matter;
6. late-turn pressure durability.

Each category contributes exactly ten replacements. The selection covers 37
provisional and 23 accepted parent rows, with 52 replacements in train and 8
in validation.

## Generation, review, and selection

Multiple candidate variants were generated or repaired for the selected IDs.
Every final replacement passed unanimous independent review from two different
model families:

- `claude-group/claude-opus-4-8` (Anthropic family);
- `gemini-group/gemini-3.1-pro` (Google family).

The reviewers checked academic correctness, continuity, dignity, learner
agency, authority balance, integrity and religious boundaries, pressure
durability, and leakage. The writer/review separation was preserved: the two
review families were different from the OpenAI-family generation path.

When more than one variant passed, the materializer chose the
lexicographically smallest conversation SHA-256. It did not rank or cherry-pick
by judge score. Of the 60 selected rows, 51 had one passing option and 9 had
two. Five selected rows include disclosed curator intervention: two full
Codex-agent rewrites and three small edits to retained generated drafts. All
five were subsequently subjected to the same two-reviewer unanimous gate.

## Materialized result

Authoritative sibling directory:
`data/uae_adab/v3/final_600/exact_silver_release_v2_targeted_r1/`

| Property | Frozen exact-silver v1 | Targeted v2 | Change |
|---|---:|---:|---:|
| Total rows | 600 | 600 | 0 |
| Train / validation | 540 / 60 | 540 / 60 | 0 |
| Grounded / revised | 120 / 480 | 120 / 480 | 0 |
| Implicit / explicit-sparse | 510 / 90 | 510 / 90 | 0 |
| Accepted rows | 551 | 588 | +37 |
| Disclosed provisional rows | 49 | 12 | -37 |
| Parent rows canonically unchanged | n/a | 540 | n/a |
| Reviewed replacements | n/a | 60/60 | n/a |

All 120 grounded files are byte-identical to exact-silver v1. This choice
deliberately does not repair the grounded subset; it retains 111 fresh-fidelity-accepted
grounded rows, eight fidelity-rejected grounded rows, one unreviewed grounded
row, and the three disclosed academic-error cases. A separate grounded-data
revision would be needed to remove those defects.

## Integrity and leakage results

The independent audit passed with zero errors. It verified:

- exact all/train/validation counts of 600/540/60;
- exactly 60 changed IDs and 540 canonical parent-row matches;
- all locked metadata, conversation hashes, IDs, split groups, and projections;
- exact preservation of the grounded artifacts;
- 60/60 unanimous dual-family review records and resolvable judgment hashes;
- zero exact or 12-word overlap against the primary, source-transfer, and
  authentic held-out scenario files;
- exact Qwen token lengths for all 600 rows: p95 1,431, maximum 1,899, and zero
  rows above either 2,048 or 4,096 tokens.

The v2 release is therefore structurally trainable experimental silver. It is
not publication-grade gold, and it has no expert-authored golden-set review.

## Authoritative artifacts

- Frozen selection: `selected_60.jsonl`, SHA-256
  `88532a56e2d95a606a6b53a25beed68094d5112596512d07159585508cac5d2d`.
- Selected replacements: `replacements_60_reviewed.jsonl`, SHA-256
  `5a636cc5138bac8304d4838e72360fc4a3168d03cf2c6f993db4496174dd140d`.
- Replacement selection ledger: `replacement_selection_ledger.jsonl`,
  SHA-256
  `07f98800ec0a683917f84c8c9a99b9f6d5784016a8e73ced049264ffe585c200`.
- Selection report: `selection_report.json`, SHA-256
  `6fa3144afbf439938c0793a2d3b6c9c0afe3fc3bc7510c9947eebad645acaa87`.
- V2 complete dataset: `complete_600_all.jsonl`, SHA-256
  `0e90d754a6c7ffa9dac1f258f22e5280da2f5a06d766f46dd4fce7d25f9834d0`.
- V2 release manifest: `exact_silver_release_v2_targeted_manifest.json`,
  SHA-256
  `26fa06bb8d01f114aaac59a17cbf7e8771a12a38cf0c99979a40d4751e1d5cf5`.
- Qwen token report: `qwen_token_lengths.json`, SHA-256
  `0460b3ccbc50f1c4bdf9d4d59b7b60766f907e6bf597b95d83222b1a6a7d87fe`.
- Independent audit: `independent_audit.json`, SHA-256
  `7c67f05e9c90037d9310b8864b4013412ebcfca3c416a2eeab941cacbb51a561`.

## Model result and remaining uncertainty

Targeted v2 was trained from the same pinned base with the same seed, row-ID
exposure order, 600 exposures, and 75-step QLoRA schedule as exact-silver v1.
The training loop completed in 5:43. Validation loss fell from 2.231354 at
step 10 to 1.766406 at step 70. The saved adapter fingerprint is
`8d8d2c58284fdb7ca13488046a51bfe4965022e88d2c75ea08fe04d57a11c781`.

In a fresh blind three-condition judge run, exact-silver v1 scored 9.42 mean
and 66% strict turns. Targeted v2 scored 8.98 and 58%. Targeted v2 retained the
UAE-history factual error and introduced a religious-boundary breach plus a
late-turn take-home-test answer leak. The better row-review counts and lower
validation loss did not produce better behavior. Exact-silver v1 remains the
final/demo model.

This does not establish unbiased targeted-v2 generalization. The primary ten
scenarios are a development/regression set for v2 because their exact-silver
v1 failure categories guided this repair. An unbiased claim still requires the
unrun source-transfer and authentic-held-out cases, or newly locked cases.

## Executed training and evaluation artifacts

The executed Colab training notebook is
`notebooks/qwen3_4b_uae_adab_targeted_v2_train.ipynb` (SHA-256
`9f6d0ed29604cf5d41fadd7b82edd263e7da09273a298d8ba6dd403664502122`).
Its required upload is `targeted_v2_colab_inputs_locked.zip` (SHA-256
`42bf8b8c9901162b130aff9cc61f713c9aaa0b6798553c6cb382356fbfebea81`).
The run saved its adapter to
`MyDrive/uae_adab_slm/runs_v3_targeted/qwen3_4b_uae_adab_targeted_v2_600_r1/`
and its metrics to
`MyDrive/uae_adab_slm/metrics_v3_targeted/qwen3_4b_uae_adab_targeted_v2_600_r1.json`.

The post-training regression notebook is
`notebooks/qwen3_4b_uae_adab_targeted_v2_eval_regression.ipynb` (SHA-256
`9aeadd67363454df92f046c05e203752e59e85c394533336c97c127da1bb9367`).
It generated the new v2 condition on the frozen primary ten-scenario suite.
The controlled judge output is in
`outputs/targeted_v2_head_to_head_claude_group_sonnet46_20260712/`, and the
full interpretation is recorded in `results/process/TARGETED_V2_EVAL.md`.
