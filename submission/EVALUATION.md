# Evaluation handoff

## Behavior under test

Across a pressured multi-turn lesson, the tutor should remain academically
accurate, correct without humiliation, protect authorship and integrity, allow
respectful truth-seeking, stay within religious boundaries, and continue useful
tutoring without late-turn drift.

## Primary result

| Metric | Base Qwen with strong prompt | Exact-silver v1 tuned Qwen |
|---|---:|---:|
| Mean score /10 | 6.26 | **9.34** |
| Strict-turn pass | 18% | **64%** |
| Turn-five strict pass | 30% | **80%** |
| Fully durable conversations | 0% | **10%** |
| Hard-gate-clean turns | 62% | **96%** |

The assignment claim rests on this frozen behavior-pressure suite, not on a
single demo prompt.

## Conditions

- `base_no_prompt`: pinned Qwen base with no system prompt.
- `base_strong_prompt`: the same base with the frozen UAE Adab prompt.
- `complete_600_exact_silver`: the selected adapter with only the fixed control
  token.

All Qwen conditions use the same base revision, native chat template,
`enable_thinking=False`, deterministic decoding, 4,096-token context, and no
silent truncation. The judge does not see model or condition identities.

The primary comparison is strong-prompt base versus Complete-600. Because the
two deployments use different system messages, the measured delta belongs to
the complete deployments rather than the adapter alone.

## Files of record

- Scenarios: `evals/uae_adab/scenarios.json`
- Saved model outputs: `zeval/behavior_pressure_10_outputs/`
- Judged results: `outputs/exact_silver_behavior_claude_group_sonnet46_20260711/`
- Consolidated interpretation: `results/process/COMPARISONS.md`

## Frontier context

Saved GPT-5.6 Luna strong-prompt answers were rescored by the same blind Claude
Sonnet 4.6 judge. Luna scored 9.44/10 and 80% strict turns, while tuned Qwen
scored 9.34/10 and 64%. Luna used temperature 0.2 and a 700-token answer budget;
Qwen used deterministic decoding and 512 tokens. Treat this as supplementary
context, not the controlled primary comparison.

Claude was the judge, not a measured answer-model condition. The three-window
video includes fresh Claude outputs only as an illustration and must not claim
that Claude was beaten or formally evaluated.

## Reproduce

Use the two canonical notebooks:

1. `submission/notebooks/01_train_exact_silver_v1.ipynb`
2. `submission/notebooks/02_evaluate_exact_silver_v1.ipynb`

For interactive review, open the temporary Colab demo:
<https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb>.

After notebook 02 generates its three JSONL files, validate and score them with:

```bash
.venv/bin/python evals/uae_adab/score_saved_outputs.py \
  --input /path/base_no_prompt.jsonl \
  --input /path/base_strong_prompt.jsonl \
  --input /path/complete_600_exact_silver.jsonl \
  --judge-model YOUR_COMPATIBLE_JUDGE_MODEL \
  --output-dir outputs/reproduced_submission_eval \
  --dry-run
```

Remove `--dry-run` only after structural validation and only when a compatible
judge credential is available. The answer-generation notebook needs no judge
credential.

Do not combine the 9.34 primary result with the later 9.42 targeted-v2
development/regression rerun. They are separate judge runs.

## Limits

The primary suite has ten scenario clusters, one training seed, one
deterministic generation per Qwen condition, one external judge, and no
confidence interval. The source-transfer and authentic held-out suites remain
unrun.
