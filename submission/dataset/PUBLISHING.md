# Verify or reproduce the public Hugging Face dataset

The selected dataset is already public at
<https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600>. The published
540/60 files, manifest, limitation ledger, and card were downloaded without
credentials and matched the local release hashes exactly.

Immutable published revision:
`19567b9cf7e976cc526494defc4209d99e08ccd9`.

## Verification

1. Open the public dataset repository
   `adarshrajesh/uae-adab-tutor-600`:
   <https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600>.
2. Confirm these repository files are present:

| Local file | Hugging Face name |
|---|---|
| `submission/dataset/README.md` | `README.md` |
| `data/uae_adab/v3/final_600/exact_silver_release_v1/complete_600_train.jsonl` | `train.jsonl` |
| `data/uae_adab/v3/final_600/exact_silver_release_v1/complete_600_validation.jsonl` | `validation.jsonl` |
| `data/uae_adab/v3/final_600/exact_silver_release_v1/exact_silver_release_manifest.json` | `release_manifest.json` |
| `data/uae_adab/v3/final_600/exact_silver_release_v1/release_limitation_ledger_600.jsonl` | `release_limitation_ledger_600.jsonl` |

3. Confirm the dataset viewer reports 540 train and 60 validation records.
4. Confirm the card displays `license: other` and the mixed-rights explanation.
5. Open the repository in an incognito browser to confirm that no token or
   access request is required.

The local
[`uae_adab_hf_dataset_upload.zip`](uae_adab_hf_dataset_upload.zip) is a recovery
bundle containing the same five files under their final repository names. Do
not upload that ZIP as a single dataset file; extract it first if the public
repository ever needs to be recreated.

Do not upload raw captions, private permission documents, rejected candidates,
evaluation expected answers, or the historical synthetic dataset.
