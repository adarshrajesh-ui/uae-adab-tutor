# Final links

A URL resolving is not the same as a completed upload. The status column keeps
that distinction explicit until the final signed-out browser check.

| Deliverable | URL | Status on 2026-07-12 |
|---|---|---|
| GitHub repository | <https://github.com/adarshrajesh-ui/uae-adab-tutor> | Public; curated reviewer release is on `main` |
| Public Hugging Face dataset | <https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600> | Public; 540/60 files, manifest, ledger, and card uploaded and hash-verified |
| Public Hugging Face model | <https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b> | Public; selected adapter and reviewer-facing model card uploaded and hash-verified |
| Temporary reviewer demo | <https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb> | Public notebook and Colab launch page resolve; live GPU smoke test remains |
| Brainlift | OWNER TODO | Separate project-owner deliverable; intentionally not packaged here |
| 3–5 minute demo video | Not recorded yet | Pending |

The Colab URL is stable, but each GPU runtime and model session is temporary.
The Colab landing page can load even when the referenced GitHub file is
missing, so verify that the raw notebook URL also returns HTTP 200 after the
curated commit.

## Immutable deployment records

- Curated GitHub release commit:
  `b18063ff80fd1def6b396b229a5ba876956f987a`.
- Original adapter fingerprint:
  `c98e6ebda3e2bd8f6b885a318f8987035547477f796717c76d4846824df20ce0`
- Model commit containing the selected adapter:
  `c20d382f32810deaff2f691cdf78c0a3a4d9be59`.
- Current model-card revision:
  `5fc383edb6662c80731cbb7333f66c60a01f7090`.
- Dataset commit containing the 540/60 files and card:
  `19567b9cf7e976cc526494defc4209d99e08ccd9`.
