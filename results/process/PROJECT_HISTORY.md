# Project history and iteration log

## 1. Assignment and success criterion

The assignment was to instill one narrow learning or teaching behavior in a
small open model using supervised fine-tuning, with the dataset as the main
artifact. The required causal test was not whether the small model became
generally smarter; it was whether training data made a target behavior more
reliable than a strong prompt on the same base model.

The final target became:

> An English-language UAE academic tutor that teaches ordinary school subjects
> accurately while making dignity-preserving correction, respectful but
> truth-seeking treatment of parent and teacher authority, learner ownership,
> integrity, humility, and bounded Islamic civic conduct part of the teaching
> method across a pressured multi-turn lesson.

The failure being trained against was drift: a model may sound polite on one
turn but later become humiliating, over-deferential, academically unhelpful,
permissive about cheating, or religiously overreaching.

## 2. Target-selection iteration

The project began with five religious-pedagogy candidates: Evangelical
homeschool, Sunni/UAE adab, LDS, Catholic-classical, and Modern Orthodox
Jewish. Three related pretraining measurements must not be conflated:

| Measurement | Answer / judge | Strict result | Other result |
|---|---|---:|---|
| Initial five-target prompt test | GPT-5.5 / GPT-5.5 | Sunni 36.7% | pedagogy 40%; long-chat strict 0/5 |
| Five-target selection rerun | GPT-5.6 Luna / Luna | UAE 42% | no fully durable long conversation |
| Dedicated UAE benchmark | GPT-5.6 Luna / Luna | strong prompt 64% | mean 9.40; turn 5 70%; durable 20% |

The initial run graded 300 strong-prompt responses across all five targets. The
five-target Luna rerun used the target-selection rubric and produced the 42%
figure. The dedicated UAE run used ten five-turn scenarios and a different
benchmark rubric; because Luna answered and judged its own responses, its 64%
figure has self-evaluation bias. Those saved dedicated-run answers were later
blind-rescored by Claude at 80% strict, 9.44 mean, 100% turn-five, and 50% fully
durable. The 36.7%, 42%, 64%, and 80% values therefore describe different
runs, targets or judges; none should be presented as an unexplained revision
of the same score.

Together these runs established a residual reliability problem while showing
that a strong prompt already handled many easy turns. The user selected UAE
adab rather than the original Evangelical direction.

Evidence:

- outputs/prompt_test_20260707_141326/ANALYSIS_REPORT.md
- outputs/five_target_selection_luna_20260709/summary.json
- outputs/five_target_selection_luna_20260709/manifest.json
- evals/five_target_selection/
- evals/uae_adab/RESULTS.md
- outputs/uae_adab_luna_combined_20260709/summary.json
- outputs/luna_strong_prompt_claude_group_sonnet46_20260711/summary.json

## 3. Behavior-design iteration

The early design had two visible registers: institutional-light and
family-rich. Research and critique produced a three-layer concept:

1. shared conduct core;
2. fixed civic/safety floor;
3. tunable religious expression.

The user later rejected two separate model voices and requested one default
voice between them, leaning institutional rather than family-rich. The final
single-register data policy therefore became:

- one fixed system token: <uae_adab_tutor>default</uae_adab_tutor>;
- 85% shared implicit expression;
- 15% sparse explicit-adab expression;
- no user-facing light/rich mode;
- no fatwas, sect selection, scripture proof-texting, or replacement of
  parents, teachers, or qualified religious authorities.

## 4. Synthetic v1 dataset and smoke test

The first large release was a 600-row synthetic dual-register corpus, split
540/60 by design groups. A 20-row smoke subset was deliberately overfit first.
Its logged loss fell from roughly 3.95 to roughly 0.25 over 60 steps, proving
that the QLoRA/data/template loop worked.

The first real Qwen3-4B run used 540 training conversations, one epoch, 68
optimizer steps, rank-16 LoRA, and a T4. The user-observed final epoch table was
approximately 1.6715 training loss and 1.6637 validation loss. The run took
about 21 minutes and saved the qwen3_4b_uae_adab_v1 adapter.

This corpus was later judged inadequate as the final dataset because it was
essentially synthetic. It remains an important pipeline and behavior proof,
not the authenticity claim.

Evidence:

- data/uae_adab/release/accepted_train.jsonl
- data/uae_adab/release/accepted_validation.jsonl
- data/uae_adab/release/split_report.json
- zshots/smoke_20_metrics.json
- zshots/real_v1_metrics.json

## 5. Synthetic v1 post-training evaluation

The historical v1 adapter was evaluated at a 4,096-token context against base
Qwen with no prompt and with the frozen strong prompt. The same adapter was
invoked with light and rich register tokens.

| Condition | Mean /10 | Strict turns | Turn-5 pass | Fully durable |
|---|---:|---:|---:|---:|
| Base Qwen, no prompt | 4.96 | 20% | 30% | 0% |
| Base Qwen, strong prompt | 6.50 | 20% | 40% | 0% |
| Synthetic v1, light | 9.40 | 66% | 90% | 10% |
| Synthetic v1, rich | 9.48 | 62% | 80% | 10% |

This established that training could alter behavior, but it did not answer the
user's authenticity objection. It also motivated collapsing the two registers
into one default model. The historical package is descriptive only: the local
records do not cryptographically verify the v1 adapter identity, the training
environment did not pin the Unsloth version, and generation did not explicitly
set `enable_thinking=false`. The later
`outputs/final_five_condition/baseline_artifact_readiness.json` documented
those gaps; its proposed five-condition design was superseded by the final
four-condition exact-silver evaluation.

Evidence:

- outputs/qwen3_4b_4k_base_vs_tuned/condition_summary.csv
- outputs/qwen3_4b_4k_base_vs_tuned/summary.json

## 6. Authenticity reset and grounded-v3 definition

The user required at least 20% of the final data to incorporate actual source
substance, not merely be inspired by abstract concepts. Grounding was defined
as visibly carrying source academic content, teaching moves, or translated
lesson substance into the case, while allowing transformation into a safe
English training conversation. All 600 final conversations were nevertheless
Luna-written; the 20% claim describes traceable source provenance, not human
surface authorship or verbatim human tutoring transcripts. Within the 120,
17 cases use approved direct-quote grounding and 103 use substantive
nonverbatim source grounding.

The grounded-120 snapshot contains:

| Source family | Rows | What is incorporated |
|---|---:|---|
| MathDial | 64 | Source word problems, misconceptions, solution paths, and tutoring moves |
| ConvoLearn | 35 | Human-tutor academic trajectories and substantive teaching sequence |
| Arabic YouTube lessons | 21 | Permission-attested Arabic lesson substance translated and transformed into English tutoring |

Grounded review status:

- 108 fresh dual-accepted;
- 10 Arabic fresh-review-disputed;
- 1 ConvoLearn fidelity-disputed;
- 1 MathDial deterministic-valid but writer-unreviewed reconstruction.

Fresh exact-case fidelity review accepted 111 of the 120 grounded rows. This
reconciles with the tiers as follows: 108 were dual-accepted; among the 10
fresh-review-disputed rows, 3 passed fidelity but failed pedagogy and 7 failed
fidelity; the ConvoLearn row was the eighth fidelity rejection; and the
MathDial reconstruction was unreviewed. The independently
fresh-fidelity-accepted share is therefore 111/600, or 18.5%. Exactly 20%
remains accurate only as a mechanically traceable grounded-silver share that
openly includes the disputed and unreviewed rows above.

The user explicitly froze these as experimental silver rather than continuing
unbounded review. The 21 YouTube-derived rows are not open-license data: they
rely on the project owner's permission attestation dated 2026-07-10 and use
nonverbatim translated incorporation. That permission basis, rather than the
YouTube Standard license, is the asserted authority for noncommercial
experimental use. The underlying permission documents were not inspected or
copied into this repository, so their channel, video, and redistribution scope
must be demonstrated before public release.

## 7. Exact-silver v1 600-row release (20% grounded)

The final frozen release combined:

- 120 grounded rows;
- 480 revised synthetic rows;
- 510 shared-implicit rows;
- 90 sparse-explicit rows;
- 540 training rows;
- 60 validation rows;
- exactly 20% grounded material in train, validation, and the whole corpus.

The 480 revised rows comprise a 385-row accepted base plus a 95-row gap:

- 58 reviewed acceptances;
- 36 structurally preflighted but unreviewed rows;
- 1 prior-review-rejected fallback, disclosed explicitly.

Across all 600 rows, 551 were reviewed acceptances and 49 were provisional
silver rows. No train/validation lineage overlap or frozen-evaluation text
overlap was detected. Maximum Qwen token length was 1,899, below the 4,096
training limit.

The split gate found zero source-group/video-key overlap, not zero high-level
source-family overlap: MathDial, ConvoLearn, and Arabic YouTube all appear in
both splits. Three provisional training rows also retain reviewer-flagged
academic errors and must be disclosed:

- `case_uae_adab_default_arvid_fnbfetkhdc_03` -- unsupported feet-per-minute
  affirmation;
- `case_uae_adab_default_arvid_fnbfetkhdc_06` -- incorrectly labels ratios as
  y-values;
- `case_uae_adab_default_arvid_stz_chwha_04` -- contradicts the source on
  beetle wings.

Authoritative release:

- data/uae_adab/v3/final_600/exact_silver_release_v1/
- exact_silver_release_manifest.json
- manifest SHA-256:
  ef8b907a497444b83410bd2fdcff37c4eb3922d90e485638303b8e5342ba662b

## 8. Exact-silver v1 training ablation

Both runs started independently from:

- Qwen/Qwen3-4B-Instruct-2507;
- revision cdbee75f17c01a7cc42f958dc650907174af0554;
- 4-bit QLoRA;
- rank 16, alpha 32;
- response-only loss;
- 600 deterministic training exposures;
- 75 optimizer steps;
- 4,096-token limit.

### Complete-600

- 540 unique train rows and 60 validation rows;
- completed on an L4 after manual Colab validation changes;
- about 9 minutes 39 seconds;
- validation loss fell monotonically from 2.1964 at step 10 to 1.7373 at
  step 70;
- no visible validation overfitting;
- adapter saved as
  qwen3_4b_uae_adab_complete_600_exact_silver_v1.

The complete-run metrics file is present locally at
`zshots/qwen3_4b_uae_adab_complete_600_exact_silver_v1.json` with SHA-256
`cd683ce7...`.

### Grounded-120

- 108 unique train rows and 12 validation rows;
- rows cycled to the same 600 exposures;
- completed on an L4 after the same manual Colab validation changes;
- about 6 minutes 26 seconds;
- validation loss improved from 2.3183 to 1.9049 at step 40, then rose
  gradually to 1.9338 at step 70;
- this is mild expected overfitting from repeating the small corpus;
- adapter saved as
  qwen3_4b_uae_adab_grounded_120_exact_silver_v1.

The grounded run's exact runtime and loss values were observed in the completed
remote run and its expected metrics SHA-256 (`fe98f93a...`) is recorded in the
generation manifest, but the metrics JSON itself is not present locally. Those
figures are therefore not reproducible from the checked-in artifacts alone.

## 9. Operational corrections

Several notebook and Colab issues were found and documented:

- The checked-in
  `notebooks/qwen3_4b_uae_adab_exact_silver_ablation_v2.ipynb` still asserts
  A100-only execution even though L4 was sufficient for both completed runs.
- That checked-in notebook also still requires every row-level
  `silver_freeze.release_authority` field to be false. The release contains 58
  reviewed gap rows where this field is correctly true. In the executed Colab,
  this check was manually changed to require a boolean. The separate top-level
  manifest assertion `release_authority is False` remains correct.
- Consequently, the repository notebook is a pre-patch template, not an exact
  reproduction artifact for the successful L4 runs. The actually executed
  patched Colab notebook still needs to be exported and indexed.
- A stale same-named upload caused a complete_600_train hash mismatch. A
  verified five-file ZIP was produced to preserve exact bytes.
- Colab disconnected after the complete run became idle, but the 75/75 steps,
  SAVED line, adapter folder, and metrics file confirmed completion.
- The first final judge model identifier used an obsolete provider prefix and
  returned 403. The gateway model inventory showed the authorized identifier
  claude-group/claude-sonnet-4-6; the failed output directory was not reused.

## 10. Exact-silver v1 behavior evaluation

The frozen behavior suite contains ten conversations with five persistent
turns each. Four conditions were generated deterministically at 4,096 tokens
without truncation:

- base Qwen, no prompt;
- base Qwen, frozen strong prompt;
- grounded-120;
- complete-600.

All 40 conversations and 200 turns were blind-scored by
claude-group/claude-sonnet-4-6. Condition, model, adapter, and generation
metadata were withheld.

| Condition | Mean /10 | Strict turns | Turn-5 pass | Fully durable | Hard-gate-clean |
|---|---:|---:|---:|---:|---:|
| Base Qwen, no prompt | 4.94 | 14% | 20% | 0% | 52% |
| Base Qwen, strong prompt | 6.26 | 18% | 30% | 0% | 62% |
| Grounded-120 | 7.68 | 28% | 20% | 0% | 82% |
| Complete-600 | **9.34** | **64%** | **80%** | **10%** | **96%** |

Complete-600 therefore improved over the same strongly prompted base by:

- 46 percentage points in strict-turn passing;
- 50 points in turn-five passing;
- 34 points in hard-gate-clean turns;
- 3.08 points on the ten-point mean score.

This is the project's primary practical comparison, but not a pure adapter-only
isolation. The base condition used the frozen strong prompt (SHA-256
`2788520f...`), while each tuned condition used its adapter plus the fixed
deployment token (SHA-256 `ef744f9f...`). The result supports the complete
trained deployment over a strong prompted baseline; it does not attribute the
entire delta to the adapter alone.

The grounded-only ablation improved several safety and conduct dimensions but
did not hold late-turn behavior. The result is consistent with the broader
revised coverage helping durability beyond the grounded subset, but one seed
cannot definitively decompose coverage from every other corpus difference.

## 11. Frontier comparison

The original Luna strong-prompt run was initially judged by Luna itself and
therefore was not directly comparable. Its saved answers were re-scored with
the same blind Claude judge used for Qwen.

| Condition | Mean /10 | Strict turns | Turn-5 pass | Fully durable | Hard-gate-clean |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Luna, strong prompt | **9.44** | **80%** | **100%** | **50%** | 96% |
| Qwen3-4B complete-600 | 9.34 | 64% | 80% | 10% | 96% |

Qwen did not beat Luna overall. It came within 0.10 mean-score points and tied
hard-gate cleanliness. Qwen was slightly stronger on pressure resistance,
boundary integrity, and the delete test, while Luna was stronger on strict
durability, factual accuracy, academic usefulness, and authority balance.

This comparison still has a decoding caveat: Luna used temperature 0.2 and a
700-token answer budget, while Qwen used deterministic decoding and a
512-token budget.

## 12. Evaluation uncertainty

The exact-silver headline results come from one training seed per adapter, one
deterministic answer generation per condition, ten scenario clusters, one
external judge, and no bootstrap analysis or confidence intervals. All 200
turns were judged, but turns within a conversation are not independent samples.
The full frozen inventory has 31 five-turn scenarios: 10 primary, 20
source-transfer, and 1 authentic held-out. Only the primary 10 have been run;
the other 21 remain unrun. No source-transfer or authentic-case generalization
conclusion should be drawn yet.

## 13. Remaining exact-model failures

In the initial four-condition run, exact-silver v1 Complete-600 had zero
humiliation, authority-extreme, integrity-breach, and religious-boundary flags
across its 50 behavior turns. Its two hard-gate failures were factual:

- incorrect reasoning about a coin falling in air;
- confusion between UAE founding in 1971 and Ras Al Khaimah joining in 1972.

Other strict misses involved:

- insufficiently completing a mathematical comparison;
- failing to offer a concise peer-repair/apology phrase;
- refusing full verification after the tutor had made an error;
- overgeneralizing learner ownership into excessive withholding;
- insufficiently face-saving first-turn correction.

These categories, not the held-out wording, define the targeted v2 repair.

## 14. Targeted-v2 choice #1 data iteration

The first post-evaluation choice was to repair the data before retraining.
Exactly 60 revised rows were selected from the failure categories in section
13: 37 disclosed provisional rows and 23 previously accepted rows, split 52
train / 8 validation. Six categories received ten replacements each:

- calibrated directness;
- transparent tutor recovery;
- peer-harm repair;
- authority and respectful truth-seeking;
- factual precision;
- late-turn pressure durability.

Multiple candidate and repair rounds were produced. Each selected replacement
passed both `claude-group/claude-opus-4-8` and
`gemini-group/gemini-3.1-pro`. When two candidates passed, the materializer
used the lexicographically smallest conversation hash rather than judge-score
ranking. Fifty-one IDs had one passing option and nine had two. Five selected
rows include disclosed curator intervention: two full Codex-agent rewrites and
three small edits to generated drafts; all five then passed the same dual
review.

The result was materialized as a sibling rather than overwriting exact-silver v1:

- `data/uae_adab/v3/final_600/exact_silver_release_v2_targeted_r1/`;
- 600 total, 540 train, 60 validation;
- 120 grounded and 480 revised;
- 510 implicit and 90 explicit-sparse;
- 60 changed revised rows and 540 canonically unchanged parent rows;
- 588 accepted and 12 provisional, versus exact-silver v1's 551 and 49;
- every grounded artifact byte-identical to exact-silver v1.

An independent audit returned zero errors. It also found zero exact or
12-word overlap against all three frozen evaluation files. Exact
`Qwen/Qwen3-4B-Instruct-2507` tokenization gave p95 1,431 and maximum 1,899,
with no row over 2,048 or 4,096 tokens.

Zero textual leakage does not make every suite statistically held out for v2.
The repair categories were derived from exact-silver v1 failures on the
primary ten scenarios, so that primary suite becomes a development/regression benchmark
for the v2 iteration. It can measure whether the targeted repair moved the
known failures, but unbiased v2 generalization must use the still-unrun 20
source-transfer scenarios, the authentic held-out scenario, or new locked
cases. This caveat does not alter the already completed exact-silver v1 comparison.

Choice #1 did not repair the grounded subset. The 12 remaining provisional
rows are grounded: 10 fresh-review-disputed, one fidelity-disputed, and one
writer-unreviewed. The known three grounded academic-error cases therefore
remain in both exact-silver v1 and targeted v2. Targeted v2 was subsequently
trained and evaluated, as recorded in section 15.

Authoritative evidence:

- `data/uae_adab/v3/targeted_v2_repair_r1/selected_replacements/selection_report.json`;
- `data/uae_adab/v3/targeted_v2_repair_r1/selected_replacements/replacements_60_reviewed.jsonl`;
- `data/uae_adab/v3/final_600/exact_silver_release_v2_targeted_r1/exact_silver_release_v2_targeted_manifest.json`;
- `data/uae_adab/v3/final_600/exact_silver_release_v2_targeted_r1/qwen_token_lengths.json`;
- `data/uae_adab/v3/final_600/exact_silver_release_v2_targeted_r1/independent_audit.json`.

## 15. Targeted-v2 training, regression, and final selection

Targeted v2 was trained independently from the same pinned Qwen base as
exact-silver v1. The run preserved the base revision, seed, rank-16 QLoRA
configuration, row-ID exposure order, 600 training exposures, effective batch
size 8, and 75 optimizer steps. The GPU training loop completed in 5:43.
Validation loss fell from 2.231354 at step 10 to 1.766406 at step 70 without a
visible reversal.

The saved remote run is:

- adapter: `MyDrive/uae_adab_slm/runs_v3_targeted/qwen3_4b_uae_adab_targeted_v2_600_r1/`;
- metrics: `MyDrive/uae_adab_slm/metrics_v3_targeted/qwen3_4b_uae_adab_targeted_v2_600_r1.json`;
- adapter fingerprint: `8d8d2c58284fdb7ca13488046a51bfe4965022e88d2c75ea08fe04d57a11c781`.

The saved adapter then generated deterministic answers for all 10 primary
five-turn scenarios. Strong-prompt base Qwen, exact-silver v1, and targeted v2
were blind-scored together in a fresh run by
`claude-group/claude-sonnet-4-6`:

| Metric | Strong-prompt base | Exact-silver v1 Complete-600 | Targeted v2 |
|---|---:|---:|---:|
| Mean /10 | 6.26 | **9.42** | 8.98 |
| Strict turns | 18% | **66%** | 58% |
| Turn-5 pass | 30% | 70% | 70% |
| Fully durable conversations | 0% | **10%** | 0% |
| Hard-gate-clean turns | 64% | **98%** | 94% |

This fresh three-condition judge run is separate from the initial
four-condition result in section 10, where exact-silver v1 scored 9.34 mean and
64% strict turns. The scores should not be averaged or silently substituted.

Targeted v2 retained the UAE-history factual failure. It also introduced a
religious-boundary breach in the evolution scenario and leaked `y = 7` on the
take-home-test scenario after resisting earlier pressure. It lost 0.44 mean
points and eight strict-pass points relative to exact-silver v1.

The data audit improved from 551 accepted and 49 provisional rows to 588 and
12, but model behavior got worse. This is a completed negative data iteration,
not an unfinished run. Exact-silver v1 Complete-600 is therefore the selected
final/demo model. Targeted v2 is retained as evidence that more targeted review
and a lower validation loss do not guarantee a better behavior result.

The primary ten scenarios are a development/regression set for targeted v2
because exact-silver v1 failures on those scenarios defined the repair
categories. The 20 source-transfer scenarios and one authentic held-out
scenario remain unrun, so targeted v2 has no unbiased generalization result.

Evidence:

- `results/process/TARGETED_V2_EVAL.md`;
- `zeval/targeted_v2_primary_regression_10_outputs/`;
- `outputs/targeted_v2_head_to_head_claude_group_sonnet46_20260712/`.

## 16. Public release and reviewer-demo iteration

The selected exact-silver v1 adapter and frozen dataset were published to
public Hugging Face repositories. Anonymous live verification confirmed that
the adapter config and weights match the local frozen SHA-256 values and that
the public train, validation, manifest, and limitation-ledger files match the
selected release exactly. The immutable adapter publication commit is
`c20d382f32810deaff2f691cdf78c0a3a4d9be59`.

The final reviewer-delivery choice was implemented as a minimal temporary
Colab demo rather than a paid persistent Hugging Face Space. The demo:

- exposes only History, Input, Send, and Reset;
- preserves visible multi-turn history;
- downloads the public adapter without credentials and verifies both files;
- loads the pinned Qwen base in 4-bit NF4 on a T4-class or better GPU;
- supplies only `<uae_adab_tutor>default</uae_adab_tutor>` as system content;
- disables thinking and sampling, and refuses silent context truncation;
- creates a temporary public Gradio share URL for the active Colab session.

The local implementation is `deploy/reviewer_demo/app.py`; the canonical
notebook is `submission/notebooks/03_demo_uae_adab_tutor.ipynb`. The GitHub
repository is still empty, so the public Colab URL becomes functional only
after the curated repository is pushed.

## 17. Current completion state

Core experimental arc completed:

- target research and behavior definition;
- data generation and filtering;
- source-incorporating dataset revision;
- smoke test;
- exact-silver v1 Complete-600 and Grounded-120 training;
- targeted-v2 training as a separate data iteration;
- frozen base-versus-tuned evaluation;
- blind judge;
- error analysis.

Choice #1 is complete through model evaluation: targeted-v2 selection,
multi-round candidate generation, unanimous two-family review, deterministic
materialization, exact token audit, independent integrity audit, training, and
controlled regression scoring are done. It underperformed exact-silver v1 and
was rejected as the final model.

Reviewer deployment preparation is complete locally, and the public adapter
and dataset uploads are complete. The canonical reviewer route is the
temporary Colab/Gradio notebook, not the legacy paid Space path.

Still outstanding for the final submission:

- source-transfer and authentic-held-out evaluations, optional but useful;
- push the curated project and canonical notebook to the public GitHub
  repository so the prepared Colab URL becomes runnable;
- run one public Colab smoke chat after that push;
- final Brainlift update;
- 3–5 minute demo video.
