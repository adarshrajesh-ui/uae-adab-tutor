---
base_model: Qwen/Qwen3-4B-Instruct-2507
datasets:
  - adarshrajesh/uae-adab-tutor-600
library_name: peft
pipeline_tag: text-generation
license: other
language:
  - en
tags:
  - qwen3
  - peft
  - lora
  - education
  - tutoring
  - uae-adab
---

# UAE Adab Tutor — Qwen3-4B

This repository contains the selected Complete-600 LoRA adapter for
`Qwen/Qwen3-4B-Instruct-2507`. The base is pinned to revision
`cdbee75f17c01a7cc42f958dc650907174af0554`.

Release version: **exact-silver v1 Complete-600**.

## Behavior spec

Across a pressured multi-turn lesson, the tutor should teach the academic
content accurately, correct the specific work without humiliating the learner,
protect learner authorship and assessment integrity, allow respectful
evidence-based disagreement with adults, avoid religious rulings or sectarian
claims, and continue useful tutoring without drifting from that method.

This is a narrow behavior model. It is not intended to make Qwen generally
smarter or to replace a teacher, parent, or religious authority.

## Primary result

The frozen primary evaluation used ten five-turn scenario clusters, for 50
judged turns per condition. The same pinned Qwen base, scenarios, native chat
template, deterministic decoding, 4,096-token context, and blind Claude Sonnet
4.6 judge were used for both Qwen deployments.

| Metric | Base Qwen with strong prompt | Tuned Qwen with control token | Change |
|---|---:|---:|---:|
| Mean score /10 | 6.26 | **9.34** | +3.08 |
| Strict-turn pass | 18% | **64%** | +46 points |
| Turn-five strict pass | 30% | **80%** | +50 points |
| Hard-gate-clean turns | 62% | **96%** | +34 points |

The tuned condition used this adapter and the fixed system token
`<uae_adab_tutor>default</uae_adab_tutor>`. The base condition used a substantial
frozen behavioral prompt. This is therefore a comparison of two complete
deployments, not a pure adapter-only causal estimate.

A saved GPT-5.6 Luna strong-prompt run scored 9.44/10 and 80% strict turns when
rescored by the same blind judge. Luna used different provider decoding and a
larger answer budget, so that result is supplementary. The 4B model did not
beat the prompted frontier overall. No formal Claude answer-model benchmark was
run; any live Claude comparison in the demo is illustrative only.

## Training dataset

The public dataset is
[`adarshrajesh/uae-adab-tutor-600`](https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600).

| Component | Conversations |
|---|---:|
| Train | 540 |
| Validation | 60 |
| Source-incorporating grounded silver | 120 |
| Revised synthetic | 480 |
| MathDial within grounded silver | 64 |
| ConvoLearn within grounded silver | 35 |
| Permission-attested Arabic lesson videos within grounded silver | 21 |

The grounded rows incorporate source-specific academic substance or ordered
teaching moves. They are adapted, model-written conversations, not verbatim
gold transcripts. Fresh exact-case fidelity review passed 111 of the 120
grounded rows. The remaining rows and all provisional tiers are disclosed in
the dataset metadata and limitation ledger.

## Training configuration

| Setting | Value |
|---|---|
| Method | 4-bit QLoRA, response-only SFT |
| LoRA rank / alpha | 16 / 32 |
| Optimizer steps | 75 |
| Deterministic training exposures | 600 |
| Effective batch size | 8 |
| Learning rate | `2e-4` |
| Maximum sequence length | 4,096 |
| Packing | Disabled |
| Seed | 3407 |
| Training hardware | NVIDIA L4 |
| Recorded runtime | 579.27 seconds |

Validation loss fell from 2.1964 at step 10 to 1.7373 at step 70 in the saved
run. Behavioral evaluation, rather than loss alone, determined the selected
model.

## Inference contract

Use the native Qwen chat template, disable thinking, and supply the fixed
control token as the system message:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_id = "Qwen/Qwen3-4B-Instruct-2507"
base_revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
adapter_id = "adarshrajesh/uae-adab-tutor-qwen3-4b"

tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    revision=base_revision,
    torch_dtype=compute_dtype,
    quantization_config=quantization,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id).eval()

messages = [
    {"role": "system", "content": "<uae_adab_tutor>default</uae_adab_tutor>"},
    {"role": "user", "content": "I think 1/8 is larger than 1/6. Is that right?"},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, max_new_tokens=512)
print(tokenizer.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

The recorded evaluation refused silent truncation and required the prompt plus
the 512-token generation budget to fit within 4,096 tokens.

## Demo and reproduction

The reviewer path uses a temporary L4 or A100 Colab session. Public entry
points:

- GitHub: <https://github.com/adarshrajesh-ui/uae-adab-tutor>
- Temporary Colab demo: <https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb>
- Training notebook: `submission/notebooks/01_train_exact_silver_v1.ipynb`
- Evaluation notebook: `submission/notebooks/02_evaluate_exact_silver_v1.ipynb`

For the video, the temporary Colab output is shown beside two fresh Claude
windows, one question-only and one strongly prompted. That live Claude view is
not presented as a measured benchmark.

## Limitations and use boundary

- This is experimental silver, not publication-grade gold.
- The primary result comes from one training seed, one deterministic generation
  per Qwen condition, ten scenario clusters, and one external judge.
- The selected run still made two material factual errors in 50 judged turns.
- The separate source-transfer and authentic held-out suites were not run.
- This model is not approved as a child-facing production service and is not a
  religious authority.
- The dataset is mixed-rights and noncommercial. The 21 Arabic-video cases rely
  on the project owner's permission attestation for nonverbatim derivation and
  redistribution. Confirm the permission scope before distributing weights for
  another use.

## Frozen adapter identity

| File | SHA-256 |
|---|---|
| `adapter_config.json` | `e4955339b515d0af75922589d3633734faedf3bd34fb7a00bae6d47556e8fbf6` |
| `adapter_model.safetensors` | `7bd98762834e2d67e8c47cfc6f345a484c6f2c3f87a77e51d19b5f15fd790327` |

Original two-file directory fingerprint:
`c98e6ebda3e2bd8f6b885a318f8987035547477f796717c76d4846824df20ce0`.
