# Submission checklist

## Public artifacts

- [x] Fixed GitHub, model, dataset, and temporary Colab destinations are
  recorded in `LINKS.md`, and their public repository shells resolve.
- [x] GitHub contains the curated project commit rather than an empty shell.
- [x] Dataset opens at <https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600>
  and contains the actual card and files, not only `.gitattributes`.
- [x] The dataset viewer reports 540 train plus 60 validation rows.
- [x] Model opens at <https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b>
  and contains the exact selected adapter/model files and public card, not
  only `.gitattributes`.
- [x] Published model files reproduce adapter fingerprint
  `c98e6ebda3e2bd8f6b885a318f8987035547477f796717c76d4846824df20ce0`.
- [x] The dataset card states 120/480 provenance, source families, mixed rights,
  synthetic generation, review tiers, and limitations.
- [x] Canonical notebooks contain no credentials, personal Drive inputs, saved
  outputs, or unresolved placeholders.

## Temporary Colab demo

- [x] The Colab launch link opens
  `submission/notebooks/03_demo_uae_adab_tutor.ipynb`.
- [ ] An L4 or A100 runtime loads the pinned base and expected adapter
  fingerprint.
- [ ] A five-turn smoke test preserves history, accurate correction, dignity,
  assessment integrity, religious boundaries, and reset behavior.
- [ ] The runtime is disconnected after review or recording so it does not keep
  consuming Colab compute.

## Curated GitHub release gate

- [x] `python scripts/validate_public_submission.py` exits successfully.
- [x] `python scripts/build_public_submission_release.py --check` confirms the
  reviewer ZIP and manifest are reproducible.
- [x] `python -m pytest -q tests/test_public_submission_release.py` passes.
- [x] Canonical reviewer notebooks contain no credentials, personal local
  paths, saved outputs, or unresolved reviewer-path placeholders.
- [x] The candidate Git surface contains no `.env`, token, private key, model
  weight, raw transcript/caption, scraped payload, rejected candidate,
  reviewer trace, screenshot, or local output directory.
- [x] Every relative Markdown link resolves and the optional network check
  verifies the fixed public URLs and populated Hugging Face repositories.

## Demo video

- [ ] Video is between 3:00 and 5:00.
- [ ] Three windows are visible: Claude question-only, Claude strong prompt, and
  tuned Qwen in Colab.
- [ ] Both Claude windows are fresh temporary chats with memory, projects,
  personalization, search, and tools disabled.
- [ ] All three conditions receive identical user turns in the same order.
- [ ] Fine-tuned Qwen is labeled “fixed control token only,” not unprompted.
- [ ] Responses are not edited or selectively regenerated during recording.
- [ ] The video shows the formal base-Qwen-versus-tuned result table.
- [ ] The narration says the live Claude comparison is illustrative, not a
  formal benchmark.
- [ ] The conclusion states that saved prompted Luna remained stronger overall
  and that the measured win is small-model behavioral reliability over the same
  strongly prompted Qwen base.

## Final delivery

- [ ] Brainlift has been reviewed separately by the project owner.
- [ ] Every public link in `LINKS.md` opens in an incognito browser.
- [ ] No API key, Hugging Face token, raw restricted source, private permission
  record, or `.env` file is present in the repository or upload bundles.
- [x] The real model and dataset commit SHAs are recorded in `LINKS.md`.
