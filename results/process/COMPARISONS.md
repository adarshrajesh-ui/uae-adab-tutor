# Evaluation comparisons

## Primary assignment comparison (not adapter-only)

The strongest completed assignment comparison keeps the Qwen base and pinned
revision, frozen scenarios, chat template, deterministic decoding, and blind
judge fixed. It compares a base Qwen deployment using the frozen strong prompt
with the exact-silver v1 Complete-600 deployment using the trained adapter and
its fixed
`<uae_adab_tutor>default</uae_adab_tutor>` system token.

This is **not** a pure adapter-only causal isolation: the strong-prompt system
message has SHA-256 `2788520f...`, while the tuned deployment token has SHA-256
`ef744f9f...`. The comparison answers the assignment's practical question --
whether the trained specialist deployment is more reliable than the same base
model with a strong prompt -- but the observed delta belongs to the complete
deployment, not to the adapter alone.

| Metric | Base Qwen, strong prompt | Exact-silver v1 Complete-600 | Delta |
|---|---:|---:|---:|
| Mean score /10 | 6.26 | 9.34 | +3.08 |
| Strict-turn pass | 18% | 64% | +46 points |
| Turn-5 strict pass | 30% | 80% | +50 points |
| Fully durable conversations | 0% | 10% | +10 points |
| Hard-gate-clean turns | 62% | 96% | +34 points |
| Delete-test pass | 78% | 100% | +22 points |
| Perfect academic usefulness | 48% | 74% | +26 points |
| Perfect adab as method | 28% | 78% | +50 points |
| Perfect authority balance | 66% | 94% | +28 points |
| Perfect boundary/integrity | 52% | 96% | +44 points |
| Perfect pressure resistance | 50% | 96% | +46 points |

Conclusion: on this frozen benchmark and single run, the exact-silver v1
Complete-600 deployment passed the project's fine-tuning litmus test. The
strongly prompted version of the same base model did not reliably produce the
target behavior.

## Grounded-only ablation

| Metric | Strong-prompt base | Exact-silver v1 Grounded-120 | Exact-silver v1 Complete-600 |
|---|---:|---:|---:|
| Mean /10 | 6.26 | 7.68 | 9.34 |
| Strict turns | 18% | 28% | 64% |
| Turn-5 pass | 30% | 20% | 80% |
| Hard-gate-clean | 62% | 82% | 96% |

The two tuned conditions used the same deployment token and training exposure;
they were independently trained from the same pinned base. Grounded-only data
improved conduct and safety but showed mild validation overfitting and did not
hold late-turn durability in this run, which is consistent with insufficient
scenario breadth. The exact-silver v1 complete mix performed much better.
Because each condition used only one training seed, this ablation is strong
directional evidence rather than a variance estimate.

## Historical synthetic v1

| Condition | Mean /10 | Strict turns | Turn-5 pass |
|---|---:|---:|---:|
| Historical strong-prompt base | 6.50 | 20% | 40% |
| Synthetic v1 light | 9.40 | 66% | 90% |
| Synthetic v1 rich | 9.48 | 62% | 80% |
| Exact-silver v1 Complete-600 | 9.34 | 64% | 80% |

The historical and exact evaluations used the same frozen scenario content and
judge-prompt text, but were generated at different times and the old judge
provider identifier is no longer available. The historical package also lacks
a cryptographically verified adapter identity, did not pin the Unsloth version,
and omitted an explicit `enable_thinking=false` setting. Treat it as historical
behavior evidence, not as a fifth condition that can be mixed into the final
four-condition comparison. Behavior stayed roughly level while the dataset
became materially more defensible: 20% mechanically traceable
source-incorporating silver and one default voice rather than two synthetic
registers. All exact-silver v1 conversation surfaces remain Luna-written, and
only 111/120 grounded rows passed fresh exact-case fidelity review.

## Frontier comparison under one judge

| Metric | Luna, strong prompt | Qwen exact-silver v1 Complete-600 |
|---|---:|---:|
| Mean /10 | 9.44 | 9.34 |
| Strict turns | 80% | 64% |
| Turn-5 pass | 100% | 80% |
| Fully durable | 50% | 10% |
| Hard-gate-clean | 96% | 96% |
| Delete-test pass | 98% | 100% |

Conclusion: the small model rivals the frontier model's average constrained
behavior but does not beat its strict durability. That is supplementary
context, not the assignment's primary success criterion.

## Dataset iteration comparison

Choice #1 first produced the targeted-v2 sibling dataset, then a separately
trained adapter. The dataset audit improved, but the controlled behavior result
did not. Exact-silver v1 remains the selected final model.

| Property | Frozen exact-silver v1 | Targeted v2 sibling |
|---|---:|---:|
| Rows | 600 | 600 |
| Grounded / revised | 120 / 480 | 120 / 480 |
| Train / validation | 540 / 60 | 540 / 60 |
| Accepted / provisional | 551 / 49 | 588 / 12 |
| Rows changed from exact-silver v1 | n/a | 60 revised rows |
| Grounded rows changed | n/a | 0 |
| New replacements with dual-family unanimous review | n/a | 60/60 |
| Frozen-eval exact or 12-word overlap | 0 | 0 |
| Exact Qwen maximum tokens | 1,899 | 1,899 |
| Trained and evaluated | Yes | Yes, on primary regression suite |

The data-quality change is real and auditable: 37 provisional revised rows
were removed, and all 60 targeted replacements passed independent Anthropic-
and Google-family review. That did not translate into a better model.

## Targeted-v2 controlled regression

Strong-prompt base Qwen, exact-silver v1, and targeted v2 were blind-scored
together in a fresh judge run on the primary ten-scenario suite.

| Metric | Strong-prompt base | Exact-silver v1 Complete-600 | Targeted v2 |
|---|---:|---:|---:|
| Mean /10 | 6.26 | **9.42** | 8.98 |
| Strict turns | 18% | **66%** | 58% |
| Turn-5 pass | 30% | 70% | 70% |
| Fully durable | 0% | **10%** | 0% |
| Hard-gate-clean | 64% | **98%** | 94% |
| Delete-test pass | 74% | **100%** | 96% |

Targeted v2 still beat the strong-prompt base, but it underperformed
exact-silver v1 and introduced one religious-boundary breach plus one late-turn
academic-integrity breach. Exact-silver v1 remains the final/demo adapter. This
is the clearest project example of why training loss and review counts cannot
replace behavior evaluation.

Because v2 repair categories were derived from exact-silver v1 failures on the
primary ten scenarios, the completed v2 rerun there is a development/regression
comparison, not an unbiased held-out result. Any v2 generalization should be
reported on the unrun source-transfer/authentic suites or newly locked cases.

## Interpretation limits

The headline evaluation used one training seed per tuned condition, one
deterministic generation per condition, ten scenario clusters with five turns
each, one external judge, and no confidence interval or bootstrap analysis.
These facts limit precision even though all 200 turns were scored. The separate
20-scenario source-transfer suite and the one authentic held-out scenario have
not been run, so no source-transfer or authentic-case generalization claim is
supported yet.
