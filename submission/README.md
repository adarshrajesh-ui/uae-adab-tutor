# UAE Adab Tutor submission

This is the reviewer-facing entry point for exact-silver v1 Complete-600, a
rank-16 QLoRA adapter for the pinned `Qwen/Qwen3-4B-Instruct-2507` base.

## 1. Behavior spec

Across a pressured multi-turn lesson, the tutor should teach the academic
content accurately, correct the specific work without humiliating the learner,
protect learner authorship and assessment integrity, allow respectful
evidence-based disagreement with adults, avoid religious rulings or sectarian
claims, and continue useful tutoring without drifting from that method.

## 2. Primary result

| Metric | Base Qwen with strong prompt | Exact-silver v1 tuned Qwen | Change |
|---|---:|---:|---:|
| Mean score /10 | 6.26 | **9.34** | +3.08 |
| Strict-turn pass | 18% | **64%** | +46 points |
| Turn-five strict pass | 30% | **80%** | +50 points |
| Hard-gate-clean turns | 62% | **96%** | +34 points |

The frozen primary suite contains ten persistent five-turn scenarios. Both
Qwen conditions used the same pinned base revision, scenarios, native chat
template, deterministic decoding, 4,096-token context, and blind Claude Sonnet
4.6 judge. The base used a substantial behavioral prompt; tuned Qwen used the
adapter and its fixed control token.

## 3. Dataset composition

- 600 conversations, split 540 train and 60 validation.
- 120 source-incorporating grounded-silver conversations.
- 480 revised synthetic conversations.
- Grounded sources: 64 MathDial, 35 ConvoLearn, and 21 permission-attested
  Arabic lesson-video cases.
- Fresh exact-case fidelity review passed 111 of the 120 grounded rows.

The public card explains what grounded means, identifies the synthetic share,
and preserves row-level rights and review limitations.

## 4. Demo and reproduction

Public artifacts:

- GitHub: <https://github.com/adarshrajesh-ui/uae-adab-tutor>
- Model: <https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b>
- Dataset: <https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600>
- Temporary Colab demo: <https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb>

Open the temporary Colab on an L4 or A100, verify the adapter fingerprint, and
keep the runtime active only while testing or recording.

For the video, arrange three windows side by side:

1. Claude question-only in a fresh temporary chat.
2. Claude with the frozen strong prompt in a separate fresh temporary chat.
3. Fine-tuned Qwen in the temporary Colab, using only its fixed control token.

Send the same user turns to all three windows in the same order. The Claude
comparison is a live illustration. The formal evidence remains the frozen
base-Qwen-versus-tuned-Qwen result.

Label the tuned condition **fixed control token only**. Do not call it
unprompted.

Reproduction files:

- [`uae_adab_canonical_colabs.zip`](uae_adab_canonical_colabs.zip) bundles all
  three clean notebooks for download.
- [`notebooks/01_train_exact_silver_v1.ipynb`](notebooks/01_train_exact_silver_v1.ipynb)
  reproduces the selected training run.
- [`notebooks/02_evaluate_exact_silver_v1.ipynb`](notebooks/02_evaluate_exact_silver_v1.ipynb)
  regenerates the three Qwen conditions for blind scoring.
- [`EVALUATION.md`](EVALUATION.md) records the exact evaluation contract.
- [`video/DEMO_SCRIPT.md`](video/DEMO_SCRIPT.md) gives the recording sequence.

## 5. Limitations

- This is time-boxed experimental silver, not publication-grade gold.
- The primary result uses one seed, one deterministic generation per Qwen
  condition, ten scenario clusters, and one external judge.
- The tuned run still made two material factual errors.
- Source-transfer and authentic held-out evaluation remain incomplete.
- This is not a production child-facing service or a religious authority.
- The public dataset is mixed-rights and noncommercial.

A saved GPT-5.6 Luna strong-prompt run scored 9.44/10 and 80% strict turns when
rescored by the same judge. That comparison used different provider decoding
and is supplementary. No formal Claude answer-model benchmark was run.

The Brainlift remains a separate owner-controlled deliverable and was not
edited during this documentation pass.
