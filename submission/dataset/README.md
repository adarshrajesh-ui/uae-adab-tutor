---
pretty_name: UAE Adab Tutor 600
license: other
language:
  - en
task_categories:
  - text-generation
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: validation.jsonl
---

# UAE Adab Tutor 600

This is the 600-conversation supervised fine-tuning dataset used for
[`adarshrajesh/uae-adab-tutor-qwen3-4b`](https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b).

Release version: **exact-silver v1 Complete-600**.

## Behavior spec

Across a pressured multi-turn lesson, the tutor should teach the academic
content accurately, correct the specific work without humiliating the learner,
protect learner authorship and assessment integrity, allow respectful
evidence-based disagreement with adults, avoid religious rulings or sectarian
claims, and continue useful tutoring without drifting from that method.

The dataset targets durable teaching conduct, not religious vocabulary added
to otherwise ordinary answers.

## Result produced with this dataset

On the frozen ten-conversation, fifty-turn behavior-pressure suite, the selected
Qwen3-4B deployment improved over the same strongly prompted base:

| Metric | Base Qwen with strong prompt | Tuned Qwen with control token |
|---|---:|---:|
| Mean score /10 | 6.26 | **9.34** |
| Strict-turn pass | 18% | **64%** |
| Turn-five strict pass | 30% | **80%** |
| Hard-gate-clean turns | 62% | **96%** |

Both Qwen conditions used the same pinned base revision, frozen scenarios,
native chat template, deterministic decoding, and blind judge. The base used a
substantial behavioral prompt; the tuned model used the adapter and its fixed
control token. These numbers describe the complete deployments, not the
dataset in isolation.

## Composition

| Component | Total | Train | Validation |
|---|---:|---:|---:|
| All conversations | 600 | 540 | 60 |
| Source-incorporating grounded silver | 120 | 108 | 12 |
| Revised synthetic | 480 | 432 | 48 |

Grounded source families:

| Source family | Cases | What is incorporated |
|---|---:|---|
| MathDial | 64 | Source-specific math problem substance, misconceptions, and ordered tutor moves |
| ConvoLearn | 35 | Source-specific academic trajectory and multi-turn tutor moves |
| Permission-attested Arabic YouTube lesson videos | 21 | Nonverbatim English adaptations of source-specific academic substance and teaching moves |

All final dialogue surfaces were written or revised by a frontier teacher
model. “Grounded” does not mean copied transcript text or human-written gold.
It means that source-specific substance or an ordered source interaction is
traceable in the released conversation. Fresh exact-case fidelity review passed
111 of the 120 grounded rows. The other nine remain disclosed rather than
silently relabeled as gold.

The 480 revised rows are project-generated scenarios designed to cover
correction under pressure, learner ownership, integrity boundaries, respectful
truth-seeking, tutor error recovery, and late-turn drift. They are synthetic.

## Split and leakage controls

The split was assigned by groups, not by shuffling individual rows:

- grounded rows were grouped by source record, with all segments from the same
  Arabic video kept together;
- revised rows were grouped by subject, scenario family, and learning
  objective;
- no row ID or assigned split group crosses train and validation;
- the frozen evaluation scenarios remain external and had no exact or 12-word
  overlap with a training conversation in the release audit.

The validation split is for training diagnostics. The frozen behavior suite is
the external test used for the headline result.

## Row format

Each JSONL row contains a multi-turn `messages` list plus identifiers, split and
group fields, provenance, quality status, and integrity hashes. The first
message is always:

```json
{"role": "system", "content": "<uae_adab_tutor>default</uae_adab_tutor>"}
```

The token selects the learned tutor mode. It is not a prose behavioral prompt.

Minimal loading example:

```python
from datasets import load_dataset

dataset = load_dataset("adarshrajesh/uae-adab-tutor-600")
print(dataset["train"][0]["messages"])
```

## Sources, rights, and redistribution

This is a mixed-rights, noncommercial experimental release. `license: other`
is intentional.

- MathDial adaptations retain CC BY-SA 4.0 attribution and share-alike
  obligations.
- ConvoLearn source material is attributed under its MIT dataset license.
- The 21 Arabic lesson adaptations rely on project-owner-attested permission
  for transcript access, translation, nonverbatim derivation, noncommercial
  training, and redistribution of the resulting records. This does not convert
  YouTube Standard into an open license or permit verbatim transcript reuse.
- Revised synthetic rows are project-generated, but their inclusion does not
  remove restrictions attached to grounded components.

Each grounded row records its source URL, source revision, evidence locator,
attribution, rights basis, and applicable license or permission scope. Raw
captions and private permission documents are not included. Commercial use,
raw-caption redistribution, or relicensing requires a separate rights review.

## Training configuration of record

- Base: `Qwen/Qwen3-4B-Instruct-2507`
- Base revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- QLoRA rank / alpha: 16 / 32
- Effective batch size: 8
- Optimizer steps / deterministic exposures: 75 / 600
- Learning rate: `2e-4`
- Response-only loss, no packing, no row truncation

## Demo and reproduction

- GitHub: <https://github.com/adarshrajesh-ui/uae-adab-tutor>
- Model: <https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b>
- Temporary Colab demo: <https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb>

The GitHub repository includes the exact training and evaluation notebooks. The
temporary Colab is the live review path and exists only for the active runtime.
Live Claude windows shown in the video are illustrative and are not part of the
measured dataset result.

## Limitations

- This is time-boxed experimental silver, not a validated curriculum or 600
  reviewed gold conversations.
- The release ledger contains expedited, disputed, and structurally preflighted
  tiers. Keep those fields when making subsets.
- No domain-expert human golden set was available.
- The model trained on this dataset still made two material factual errors in
  the primary 50-turn run.
- The source-transfer and authentic held-out suites were not completed.
- The dataset is not a production child-safety corpus and does not provide
  religious advice.

## Exact files

| File | Rows | SHA-256 |
|---|---:|---|
| `train.jsonl` | 540 | `990cdc7ca494a4e12efa1cc7a739030ef1412e1dd1a350f91b1d453e49a756a0` |
| `validation.jsonl` | 60 | `cf10c2e316c13da2e2ebf1c1edee18ebfd3784085440a00b8a8c31e71fa0ec1c` |
| `release_manifest.json` | n/a | `ef8b907a497444b83410bd2fdcff37c4eb3922d90e485638303b8e5342ba662b` |
