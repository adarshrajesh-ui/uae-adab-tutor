# UAE Adab Tutor: behavior from data

This project fine-tunes Qwen3-4B into a narrow academic tutor whose teaching
method remains accurate, dignifying, integrity-preserving, and respectful under
multi-turn pressure.

## Behavior spec

Across a pressured multi-turn lesson, the tutor should teach the academic
content accurately, correct the specific work without humiliating the learner,
protect learner authorship and assessment integrity, allow respectful
evidence-based disagreement with adults, avoid religious rulings or sectarian
claims, and continue useful tutoring without drifting from that method.

## Primary result

On the frozen ten-conversation, fifty-turn behavior-pressure suite:

| Metric | Base Qwen with strong prompt | Exact-silver v1 tuned Qwen |
|---|---:|---:|
| Mean score /10 | 6.26 | **9.34** |
| Strict-turn pass | 18% | **64%** |
| Turn-five strict pass | 30% | **80%** |
| Hard-gate-clean turns | 62% | **96%** |

In short, mean score rose from **6.26 to 9.34**, while strict-turn passage rose
from **18% to 64%**.

Both conditions used the same pinned Qwen base, frozen scenarios, native chat
template, deterministic decoding, 4,096-token context, and blind judge. The
base received a substantial behavioral prompt. The tuned deployment received
the adapter plus its fixed control token. This establishes a deployment-level
fine-tuning win, not a pure adapter-only causal estimate.

## Dataset

The selected dataset contains 600 multi-turn conversations:

| Component | Count |
|---|---:|
| Train / validation | 540 / 60 |
| Source-incorporating grounded silver | 120 |
| Revised synthetic | 480 |
| Grounded MathDial / ConvoLearn / Arabic lesson cases | 64 / 35 / 21 |

The grounded rows directly incorporate source-specific academic substance or
ordered teaching moves. They are adapted model-written conversations, not
verbatim gold transcripts. Rights, source locators, review tiers, and exact
hashes remain attached to the public release.

## Try it and reproduce it

- GitHub: <https://github.com/adarshrajesh-ui/uae-adab-tutor>
- Model adapter: <https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b>
- Dataset: <https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600>
- Temporary Colab demo: <https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb>

The demo runs only inside an active L4 or A100 Colab session. For the video,
use three synchronized browser windows: Claude question-only, Claude with the
frozen strong prompt, and tuned Qwen in Colab. The live Claude windows are
illustrative, not a measured benchmark.

Training and evaluation reproduction notebooks are under
[`submission/notebooks/`](submission/notebooks/). Start with the reviewer guide
at [`submission/README.md`](submission/README.md).

## Limitations

This is experimental silver, not a production child-facing service or a
religious authority. The primary result uses one training seed, ten scenario
clusters, one deterministic generation per Qwen condition, and one external
judge. The tuned run still made two material factual errors. The separate
source-transfer and authentic held-out suites were not completed.

A saved GPT-5.6 Luna strong-prompt run remained stronger overall at 9.44/10 and
80% strict turns when rescored by the same judge. No formal Claude answer-model
benchmark was run.
