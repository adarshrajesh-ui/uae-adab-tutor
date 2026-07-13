# Canonical Colab notebooks

The reviewer path uses three notebooks. The first two reproduce the measured
training and evaluation. The third opens a temporary live demo from the public
adapter.

## 01: train exact-silver v1

`01_train_exact_silver_v1.ipynb`

- Contains the exact locked input ZIP inside the notebook.
- Requires an L4 or A100 and a runtime restart after installation.
- Validates all five release files and trains only Complete-600.
- Uses 600 deterministic exposures, 75 steps, rank-16 QLoRA, response-only
  loss, and the pinned base revision.
- Does not require Drive. Set `SAVE_TO_DRIVE=True` only if notebook 02 should
  find the adapter automatically in a later session.

Expected training time is about 6 to 12 minutes on an L4 or A100, excluding
model and package downloads. Colab availability and network speed vary.

## 02: evaluate exact-silver v1

`02_evaluate_exact_silver_v1.ipynb`

- Contains the frozen ten-scenario suite inside the notebook.
- Loads the Complete-600 adapter and metrics created by notebook 01.
- Generates no-prompt base, strong-prompt base, and tuned outputs with
  deterministic 4K no-truncation inference.
- Packages three JSONL files and a generation manifest for blind scoring.

Set `USE_DRIVE=True` when notebook 01 saved to Drive. Otherwise upload the
adapter-and-metrics ZIP produced by notebook 01.

The notebook contains no paid judge credential. The saved judged result is
checked into the project at the paths in `../EVALUATION.md`; a reviewer with a
compatible judge endpoint can rerun `evals/uae_adab/score_saved_outputs.py`.

## 03: temporary live demo

`03_demo_uae_adab_tutor.ipynb`

Open directly in Colab:
<https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb>.

- Loads the public adapter
  `adarshrajesh/uae-adab-tutor-qwen3-4b` on an L4 or A100.
- Verifies the pinned base, fixed control token, and adapter identity.
- Provides a resettable multi-turn chat for review and recording.
- Exists only while the Colab runtime is active. Disconnect the runtime after
  use.

Only the three notebooks above are part of the current reviewer path. The older
`03_deploy_exact_silver_v1.ipynb` is retained as historical material and should
not be used for submission.
