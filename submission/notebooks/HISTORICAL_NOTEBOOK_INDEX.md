# Historical notebook index

The project-level `notebooks/` directory preserves the complete experimental
history. Do not run these as the selected final reproduction path:

- `qwen3_4b_uae_adab_training.ipynb` and `qwen3_4b_uae_adab_eval_4k.ipynb`:
  historical synthetic dual-register v1.
- `qwen3_4b_uae_adab_unified_500*.ipynb`: expedited intermediate 500-row run.
- `qwen3_4b_uae_adab_v2_*.ipynb`: intermediate blended-v2 experiments.
- `qwen3_4b_uae_adab_exact_silver_ablation*.ipynb`: original two-adapter
  grounded-120 versus Complete-600 experiment, superseded for graders by
  canonical notebook 01.
- `qwen3_4b_uae_adab_targeted_v2_*.ipynb`: the documented negative data
  iteration; its adapter underperformed exact-silver v1 and is not selected.
- `qwen3_4b_uae_adab_exact_silver_eval_transfer.ipynb`: prepared source-transfer
  evaluation that was not completed and supports no final claim.
- `submission/notebooks/03_deploy_exact_silver_v1.ipynb`: earlier paid-Space
  deployment utility. It is retained locally for provenance but excluded from
  the canonical manifest and reviewer ZIP; use `03_demo_uae_adab_tutor.ipynb`
  for the temporary public reviewer path.

The canonical manifest names exactly three notebooks: training, evaluation,
and the temporary reviewer demo.
