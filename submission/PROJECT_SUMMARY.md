# Project summary

## Behavior spec

Across a pressured multi-turn lesson, the UAE Adab Tutor should teach the
academic content accurately, correct the specific work without humiliating the
learner, protect learner authorship and assessment integrity, allow respectful
evidence-based disagreement with adults, avoid religious rulings or sectarian
claims, and continue useful tutoring without drifting from that method.

Fine-tuning is justified only if this behavior is more reliable than the same
base model under a strong prompt.

## Primary result

| Metric | Base Qwen with strong prompt | Exact-silver v1 tuned Qwen | Change |
|---|---:|---:|---:|
| Mean score /10 | 6.26 | **9.34** | +3.08 |
| Strict-turn pass | 18% | **64%** | +46 points |
| Turn-five strict pass | 30% | **80%** | +50 points |
| Hard-gate-clean turns | 62% | **96%** | +34 points |
| Delete-test pass | 78% | **100%** | +22 points |

The frozen primary suite contains ten five-turn scenario clusters. Both Qwen
conditions used the same pinned base revision, frozen scenarios, native chat
template, deterministic decoding, 4,096-token context, and blind judge. The
base used a substantial behavioral prompt. Tuned Qwen used the adapter plus the
fixed control token `<uae_adab_tutor>default</uae_adab_tutor>`. The result is a
comparison of complete deployments, not a pure adapter-only causal isolation.

## Dataset composition

| Property | Final value |
|---|---:|
| Conversations | 600 |
| Train / validation | 540 / 60 |
| Grounded silver / revised synthetic | 120 / 480 |
| Grounded source families | 64 MathDial, 35 ConvoLearn, 21 Arabic lessons |
| Fresh grounded fidelity passes | 111 / 120 |
| Base model | Qwen3-4B-Instruct-2507 |
| QLoRA steps / rank | 75 / 16 |

“Grounded” means a case directly incorporates source-specific academic
substance or ordered teaching moves. It does not mean copied transcript text or
independently accepted human gold. All final conversation surfaces were written
or revised by a frontier teacher model. This remains experimental silver.

## Demo and reproduction

- GitHub: <https://github.com/adarshrajesh-ui/uae-adab-tutor>
- Model: <https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b>
- Dataset: <https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600>
- Temporary Colab: <https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb>

The video uses three synchronized windows: Claude question-only, Claude with
the frozen strong prompt, and tuned Qwen in the temporary Colab. Fresh Claude
outputs are illustrative and are not substituted for the formal evaluation.

## Frontier context

Saved GPT-5.6 Luna strong-prompt answers scored 9.44/10 and 80% strict turns
when rescored by the same blind Claude judge. Tuned Qwen scored 9.34/10 and 64%.
The small model approached Luna's mean score but did not beat its strict
durability. Luna also used different provider decoding and a larger answer
budget, so this is supplementary context.

Claude served as the blind judge in the primary run. Claude was not formally
benchmarked as an answer model. Any fresh Claude run in the video is a live
illustration only.

## Limitations

- One training seed and one deterministic generation per Qwen condition.
- Ten scenario clusters and fifty judged turns in the primary suite.
- One external judge and no confidence interval.
- Two material factual errors remained in the selected run.
- Source-transfer and authentic-held-out suites were not completed.
- The model is not approved as a child-facing production service and is not a
  religious authority.
