# Authoritative artifact index

## Specification, research, and frozen evaluations

- `PROJECT_SPEC.md`
- `research/data-sources.md`
- `evals/uae_adab/scenarios.json` -- 10-scenario primary behavior suite,
  SHA-256 `bf7932483942f33555c98649aaea425ff8127d848cf22c2637ed44f3636b08ac`
- `evals/uae_adab/source_transfer_scenarios_v0.json` -- 20-scenario frozen
  transfer suite, SHA-256
  `45f7095b64304b7589b7e9687712f01d05909d49bdc2d562f49d0b08328e1517`
- `evals/uae_adab/authentic_heldout_scenarios.json` -- separate authentic
  held-out scenario, SHA-256
  `f1544dbe5798156fb9a8b737cf5ded8d7de011f8a716694a12097bf6f35d11dc`

`brainlift.md` is currently a historical/incomplete draft: it describes the
old all-synthetic dual-register corpus and historical synthetic-v1 evaluation,
not the exact-silver release and final results. Do not cite it as the final account until it is
updated. `modelBrainlift.md` is an untouched formatting example and is not a
project artifact to edit.

## Frozen exact-silver v1 dataset (20% grounded)

Authoritative directory:
`data/uae_adab/v3/final_600/exact_silver_release_v1/`

- Release manifest: `exact_silver_release_manifest.json`, SHA-256
  `ef8b907a497444b83410bd2fdcff37c4eb3922d90e485638303b8e5342ba662b`
- Complete all (600): `complete_600_all.jsonl`, SHA-256
  `b1d55906c4b22ff7ddbe84684653e9e03eaf6a8a05bf992ff79bd9199177964b`
- Complete train (540): `complete_600_train.jsonl`, SHA-256
  `990cdc7ca494a4e12efa1cc7a739030ef1412e1dd1a350f91b1d453e49a756a0`
- Complete validation (60): `complete_600_validation.jsonl`, SHA-256
  `cf10c2e316c13da2e2ebf1c1edee18ebfd3784085440a00b8a8c31e71fa0ec1c`
- Grounded all (120): `grounded_120_all.jsonl`, SHA-256
  `d558672860ee986d0b6dc8bd52a4d64442d31b72661d70d873fee948003b8c8e`
- Grounded train (108): `grounded_120_train.jsonl`, SHA-256
  `591d6466dcda9b5aafdb1e19a51ba873c72a657e38de262e76f49d46bf37c19b`
- Grounded validation (12): `grounded_120_validation.jsonl`, SHA-256
  `28470b409c5db194dc97a34b637a1a87275ed3b685869fc52dca3c0f7b6f56a5`
- Row-level limitation ledger (600): `release_limitation_ledger_600.jsonl`,
  SHA-256
  `cb2705d7f29fb3924edbf9386ffc64023e23e3675bde47fef68a7896d5f9dee2`
- Locked five-input Colab ZIP: `exact_silver_colab_inputs_locked.zip`,
  SHA-256
  `2ad66a817804220bc23bf182d08aef2b95b711e90d63214d79f388c87ef47e94`

Grounded provenance lives in
`data/uae_adab/v3/final_600/grounded_120_silver_v1/`. Key evidence includes
`source_registry.json` (SHA-256
`38d12fe7ce27322df79d69a378e10354d8fd7d5c53f379f040d343e6485d5da9`),
`grounding_audit.json`
(`0c9f48bdf896bfd57b30db987025d88b51571ce8d8103828ec8cca88ac7e77fe`),
`case_status_ledger_120.jsonl`
(`afa0982746a75519d2a084d23c15f89da3462a1f5e7a1d712adc2e81507b3ee1`),
and `input_manifest.json`
(`44135a566133d86e7267739c471e82716ba504da6937fa62286d8dbbd4d48623`).
The grounded composition is 64 MathDial, 35 ConvoLearn, and 21 translated
Arabic YouTube lesson cases. Of the 120, 17 use approved direct-quote grounding
and 103 use substantive nonverbatim grounding. All 600 conversation surfaces
were Luna-written; source grounding does not mean that 120 rows are verbatim
human-human tutoring transcripts.

The 21 YouTube-derived rows are not open-license training data. They rely on
the project owner's permission attestation dated 2026-07-10 and nonverbatim
translated incorporation for noncommercial use; the underlying YouTube
Standard license is not an open redistribution license. The repository's
attestations are `research/grounded_v3_quote/YOUTUBE_PERMISSION_ATTESTATION.md`
(SHA-256
`40671906f52166637c0c8083f9180f3f946fea8321f46914ce2db4a58b1e1527`)
and `research/grounded_v3_quote/USER_PERMISSION_ATTESTATION_2026-07-10.md`
(SHA-256
`833de910ba25c8af5e799191c25b5cccf488b2936394090e89c7aee5323087eb`).
Codex did not inspect or copy the underlying permission documents. Reconfirm
their channel, video, use, and distribution scope before publishing. The other
grounded sources are recorded as MathDial CC-BY-SA-4.0 (share-alike; upstream
metadata ambiguity remains) and ConvoLearn MIT (the pinned card lacks a
standalone license/copyright notice).

## Review-tier disclosure

The 600 rows contain 551 accepted rows and 49 disclosed provisional rows.
Exact tiers are:

- 108 grounded `fresh_dual_accepted`;
- 385 revised `reviewed_accepted_expedited`;
- 58 revised `reviewed_accepted`;
- 10 grounded `fresh_review_disputed`;
- 1 grounded `fidelity_disputed`;
- 1 grounded `writer_unreviewed_deterministic_valid`;
- 36 revised `unreviewed_round4_structural_preflight`;
- 1 revised `provisional_prior_review_rejected_fallback`.

The expedited base used selective rather than universal dual review. This is
an experimental silver corpus, not a publication-grade gold set.

Fresh exact-case fidelity review accepted 111 of the 120 grounded rows. The
tier reconciliation is: 108 dual-accepted; 3 of the 10 fresh-review-disputed
rows passed fidelity but failed pedagogy; the other 7 failed fidelity; the
ConvoLearn disputed row was the eighth fidelity rejection; and the MathDial
reconstruction was unreviewed. Therefore 18.5% of the full corpus is
independently fresh-fidelity-accepted. The headline 20% is a mechanically
traceable grounded-silver share that includes the disclosed disputed and
unreviewed rows; it is not a claim that all 120 independently passed fidelity
review.

The split has zero overlap by source-group/video key, not by high-level source
family. MathDial, ConvoLearn, and Arabic YouTube each appear in both train and
validation.

## Training notebooks and evidence

The repository template is
`notebooks/qwen3_4b_uae_adab_exact_silver_ablation_v2.ipynb`, with supporting
instructions in the release's `TRAIN_NOW.md`. They are **not** exact
reproduction artifacts for the successful runs: the checked-in notebook still
asserts A100-only execution and still rejects row-level
`silver_freeze.release_authority=true`, although 58 valid reviewed rows use
that value. Both successful runs used an L4 after manually changing the GPU
and row-boolean checks in Colab. The executed patched notebook has not been
exported locally.

Saved remote adapter directories:

- `MyDrive/uae_adab_slm/runs_v2_exact/qwen3_4b_uae_adab_complete_600_exact_silver_v1/`
- `MyDrive/uae_adab_slm/runs_v2_exact/qwen3_4b_uae_adab_grounded_120_exact_silver_v1/`

Adapter fingerprints recorded by final generation:

- Complete-600 combined SHA-256: `c98e6ebda3e2bd8f6b885a318f8987035547477f796717c76d4846824df20ce0`
- Grounded-120 combined SHA-256: `e53afca58485ef677d99984ae9d86f16bd3dbd50117b677bb82df425dc8e0c60`

Complete-run metrics are local at
`zshots/qwen3_4b_uae_adab_complete_600_exact_silver_v1.json`, SHA-256
`cd683ce7699e7adc29fb95e4849ecc51db4d4d04762e61a64a60bc7922b055cf`.
The grounded-run metrics JSON is missing locally; the final generation
manifest records its expected SHA-256 as
`fe98f93a7335ac6366938dd3686ae1a98817f9c722c55953b64037a87f312765`.

## Exact-silver v1 behavior evaluation

- Generated ZIP: `zeval/behavior_pressure_10_outputs.zip`, SHA-256
  `0d3309dc4c623ccfaab2fc453b21e3641127a24da2e9742ee3e69e015055735f`
- Extracted outputs: `zeval/behavior_pressure_10_outputs/`
- Generation manifest: `generation_manifest.json`, SHA-256
  `f7ad544890bbb191ec2aba938b238202391182ebac6e03bed4bd750ca2fe5741`
- Blind-score directory:
  `outputs/exact_silver_behavior_claude_group_sonnet46_20260711/`
- Main report: `REPORT.md`
- Judge manifest: `manifest.json`, SHA-256
  `bf58fbb8bd532ddb53557482f97b0d31a3eb9b36467af34d23390f26dcd314d0`
- Judge summary: `summary.json`, SHA-256
  `69e3f5ebbe5245cd483b9a0482c474723471377209cf9f57a33fa9d86e916468`
- Luna same-judge rescore directory:
  `outputs/luna_strong_prompt_claude_group_sonnet46_20260711/`
- Luna-rescore manifest: `manifest.json`, SHA-256
  `31aca8fb0b9e1ef15a00c137822c2ca9431db4dd82ec5645560c034f028c9020`
- Luna-rescore summary: `summary.json`, SHA-256
  `57662134b7e3f254fd79c11c9bebc649ac9600fbb7d373c58b9ec390b273597e`

The base strong-prompt condition used system-prompt SHA-256
`2788520ff0434f4ba2106eaa216edd2d2ad67999647006f68dcb69158b1bb077`.
The tuned conditions used deployment-token SHA-256
`ef744f9f82a8901201bf750d36229b9c3a829d42ff38b1460772f30503845a0c`.
This is why the primary result is a deployment comparison rather than a pure
adapter-only isolation.

The initial exact-silver comparison contains four conditions. The earlier
`outputs/final_five_condition/baseline_artifact_readiness.json` is a superseded
readiness audit, not a final five-condition result.

## Targeted-v2 choice #1: completed negative model iteration

Exact-silver v1 above remains the selected final/demo model. Targeted v2 was
completed as a sibling dataset, trained independently, and evaluated on the
primary development/regression suite. It underperformed exact-silver v1.

Repair directory: `data/uae_adab/v3/targeted_v2_repair_r1/`

- Frozen selection: `selected_60.jsonl`, SHA-256
  `88532a56e2d95a606a6b53a25beed68094d5112596512d07159585508cac5d2d`
- Selection manifest: `selection_manifest.json`, SHA-256
  `da801639b04114d41a4291aa4fd7945510bb7f788fca3bcfc553c27ecd947044`
- Repair feedback: `repair_feedback.json`, SHA-256
  `223af361c9b63a6a0cdac7b24ae0814ea5d68f42d77bf51f7ea59f563ab3d2a1`
- Unanimously reviewed replacements:
  `selected_replacements/replacements_60_reviewed.jsonl`, SHA-256
  `5a636cc5138bac8304d4838e72360fc4a3168d03cf2c6f993db4496174dd140d`
- Replacement selection ledger:
  `selected_replacements/replacement_selection_ledger.jsonl`, SHA-256
  `07f98800ec0a683917f84c8c9a99b9f6d5784016a8e73ced049264ffe585c200`
- Selection report: `selected_replacements/selection_report.json`, SHA-256
  `6fa3144afbf439938c0793a2d3b6c9c0afe3fc3bc7510c9947eebad645acaa87`

Authoritative sibling release:
`data/uae_adab/v3/final_600/exact_silver_release_v2_targeted_r1/`

- Complete all (600): `complete_600_all.jsonl`, SHA-256
  `0e90d754a6c7ffa9dac1f258f22e5280da2f5a06d766f46dd4fce7d25f9834d0`
- Complete train (540): `complete_600_train.jsonl`, SHA-256
  `49b0570a8dac9ef654f4a792cd55717c8deec2ddb87400837c2a4a18a10e9d43`
- Complete validation (60): `complete_600_validation.jsonl`, SHA-256
  `8ffe9810cc53e8ab0231930d95ee9ebf447c23f47a75e2839b436197d49e7d27`
- Grounded all/train/validation are byte-identical to exact-silver v1, with the
  same hashes listed in the frozen-release section above.
- V2 limitation ledger (600): `release_limitation_ledger_600.jsonl`, SHA-256
  `ac211d179ac9b577142f1356ebb104baae6c75cfb6cdc0efdd1044fc94095f55`
- V2 replacement ledger (60): `targeted_replacement_ledger_60.jsonl`, SHA-256
  `07f98800ec0a683917f84c8c9a99b9f6d5784016a8e73ced049264ffe585c200`
- Release manifest: `exact_silver_release_v2_targeted_manifest.json`, SHA-256
  `26fa06bb8d01f114aaac59a17cbf7e8771a12a38cf0c99979a40d4751e1d5cf5`
- Exact Qwen token report: `qwen_token_lengths.json`, SHA-256
  `0460b3ccbc50f1c4bdf9d4d59b7b60766f907e6bf597b95d83222b1a6a7d87fe`
- Independent audit (`ok: true`, zero errors): `independent_audit.json`,
  SHA-256
  `7c67f05e9c90037d9310b8864b4013412ebcfca3c416a2eeab941cacbb51a561`
- Independent-audit script: `scripts/audit_targeted_v2_release.py`, SHA-256
  `c1bacb2de7e22fbca7969941d3b8b730744e2363bf0ff43e46f8c6dd807c45d7`
- Token-analysis script: `scripts/analyze_qwen_token_lengths.py`, SHA-256
  `23c93118f2bbcb138e7778889c59769868b3d5632f121e8dd517ad405927ce5d`
- Targeted-v2 Colab notebook:
  `notebooks/qwen3_4b_uae_adab_targeted_v2_train.ipynb`, SHA-256
  `9f6d0ed29604cf5d41fadd7b82edd263e7da09273a298d8ba6dd403664502122`
- Locked Colab input bundle: `targeted_v2_colab_inputs_locked.zip`, SHA-256
  `42bf8b8c9901162b130aff9cc61f713c9aaa0b6798553c6cb382356fbfebea81`
- Targeted-v2 primary regression notebook:
  `notebooks/qwen3_4b_uae_adab_targeted_v2_eval_regression.ipynb`, SHA-256
  `9aeadd67363454df92f046c05e203752e59e85c394533336c97c127da1bb9367`

Saved remote training artifacts:

- Adapter:
  `MyDrive/uae_adab_slm/runs_v3_targeted/qwen3_4b_uae_adab_targeted_v2_600_r1/`
- Metrics:
  `MyDrive/uae_adab_slm/metrics_v3_targeted/qwen3_4b_uae_adab_targeted_v2_600_r1.json`
- Adapter fingerprint:
  `8d8d2c58284fdb7ca13488046a51bfe4965022e88d2c75ea08fe04d57a11c781`

Saved regression and judge artifacts:

- Generated ZIP: `zeval/targeted_v2_primary_regression_10_outputs (1).zip`,
  SHA-256
  `0957476128251a20500bd3374fb17b286dc2d38aeecb0b9d5e3d55bfd1891557`
- Extracted output directory:
  `zeval/targeted_v2_primary_regression_10_outputs/`
- V2 conversations: `targeted_v2_600_r1.jsonl`, SHA-256
  `98b416489f1df8bbdb16cce9c0da8e16804d3727068c9666c28371d5fb7dab55`
- Generation manifest: `generation_manifest.json`, SHA-256
  `d912bfbeaa7aace2f8afc8eeda3a18a3afa1cd4a8dc9b687119308d8575db8fd`
- Controlled judge directory:
  `outputs/targeted_v2_head_to_head_claude_group_sonnet46_20260712/`
- Controlled judge manifest: `manifest.json`, SHA-256
  `776cf6f7b0ce8aa2be7ccc12ddc3af9fe886d50da148f71ac24912ab8a1e0075`
- Controlled summary: `summary.json`, SHA-256
  `f1be446a6b6aae5ef862cbeaaac510b62d3f62a4819c78a8444a5794d50da6a7`

The sibling replaces exactly 60 revised rows (37 provisional and 23 accepted;
52 train / 8 validation; ten per repair category) and leaves 540 parent rows
canonically unchanged. All 60 replacements passed both
`claude-group/claude-opus-4-8` and `gemini-group/gemini-3.1-pro`. The corpus now
contains 588 accepted and 12 disclosed provisional grounded rows. It preserves
120/480 grounded/revised and 510/90 implicit/explicit-sparse composition.

Replacement authorship must not be described as uniformly Luna-written: 58 of
the 60 rows retain Luna writer-model metadata and two are disclosed
Codex-agent manual rewrites; three of those 58 also received small curator
edits before unanimous review. The replacement ledger's `old_tier` is a
historical selection label and is not authoritative for every parent row. Use
the materialized provenance fields `selection_old_tier_label` and
`authoritative_parent_tier` when exact parent-tier provenance matters.

The independent audit re-resolved all 60 chosen review lineages and verified
zero exact or 12-word overlap against all three frozen suites. Exact Qwen
tokenization covers all 600 rows, with p95 1,431, maximum 1,899, and zero rows
above 2,048 or 4,096. The report is bound to the complete-file SHA-256 and to
Qwen revision `cdbee75f17c01a7cc42f958dc650907174af0554`, with tokenizer-backend,
chat-template, and special-token-map hashes recorded.

For v2, the primary ten-scenario suite is a development/regression set because
its exact-silver v1 failures defined the repair categories; byte-level freezing
and zero text overlap do not restore statistical holdout status. The unrun
source-transfer/authentic cases or new locked cases are required for an
unbiased v2 generalization claim.

The targeted-v2 notebook was executed and saved the adapter and metrics in
Drive. The repository does not contain a local copy of those weights or the
training metrics JSON. The generated regression output and controlled judge
results are present locally and bind the run to the adapter fingerprint above.

In the controlled three-condition rerun, exact-silver v1 scored 9.42 mean,
66% strict turns, and 98% hard-gate-clean turns. Targeted v2 scored 8.98, 58%,
and 94%. V2 introduced one religious-boundary breach and one late-turn
take-home-test answer leak. Exact-silver v1 was retained as the final model.

## Public endpoints and prepared deployment artifacts

Fixed reviewer-facing destinations:

- GitHub: <https://github.com/adarshrajesh-ui/uae-adab-tutor>
- Model: <https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b>
- Dataset: <https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600>
- Temporary Colab demo:
  <https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb>

Live anonymous verification on 2026-07-13 found:

- the public model repository is populated. Adapter files were published at
  immutable commit `c20d382f32810deaff2f691cdf78c0a3a4d9be59`; public hashes
  match `e4955339...e8fbf6` for `adapter_config.json` and
  `7bd98762...790327` for `adapter_model.safetensors`. The later public model
  card commit is `5fc383edb6662c80731cbb7333f66c60a01f7090`;
- the public dataset repository is populated. Data files were published at
  commit `0960927926addfdf5054c0615a0099246f1fa0a0`; public train, validation,
  release-manifest, and limitation-ledger SHA-256 values exactly match the
  frozen local files. The later public dataset-card commit is
  `19567b9cf7e976cc526494defc4209d99e08ccd9`;
- the GitHub repository still reports size zero and does not contain the demo
  notebook. Therefore the stable Colab URL is prepared but not yet runnable.

Canonical temporary reviewer artifacts:

- Runtime: `deploy/reviewer_demo/app.py`
- Runbook: `deploy/reviewer_demo/README.md`
- Notebook: `submission/notebooks/03_demo_uae_adab_tutor.ipynb`
- Contract tests: `tests/test_reviewer_demo.py`

The demo downloads the adapter anonymously at the immutable adapter commit,
verifies both file hashes, loads the pinned Qwen base in 4-bit, sends only the
fixed control token and visible chat history, refuses context truncation, and
launches a temporary Gradio share URL. It requires a T4-class or better Colab
GPU and creates no paid persistent Space.

The older optional GPU-backed Hugging Face Gradio Space remains a legacy local
path and has not been launched; it is not the canonical reviewer route.

- Deployment runbook: `deploy/README.md`
- Space website and runtime: `deploy/hf_space/`
- Adapter-publishing notebook:
  `notebooks/publish_exact_silver_v1_adapter.ipynb`
- Self-contained public deployment notebook:
  `notebooks/deploy_exact_silver_v1_to_huggingface.ipynb`
- Deterministic deployment-notebook builder:
  `scripts/build_exact_silver_hf_deploy_notebook.py`
- Adapter model-card template: `deploy/exact_silver_v1_model_card.md`
- Command-line publisher: `deploy/publish_exact_silver_v1_adapter.py`
- Contract tests: `tests/test_hf_space_deployment.py`

The runtime verifies the original `adapter_config.json` and
`adapter_model.safetensors` SHA-256 values before loading. It pins the Qwen
base revision, fixed system token, native thinking-disabled template,
4,096-token context, no-truncation policy, and deterministic 512-token answer
budget. It uses standard Transformers and PEFT rather than the exact Unsloth
evaluation runtime, so a deployed smoke regression is still required before
claiming serving-runtime equivalence.

## Known incomplete artifacts and limitations

- The source-transfer generation bundle `source_transfer_20_v0_outputs.zip` is
  absent; the 20-scenario source-transfer suite has not been run.
- The separate authentic held-out scenario has not been run.
- The frozen inventory contains 31 five-turn scenarios in total; only the 10
  primary scenarios have generated and judged outputs.
- The grounded-run metrics JSON and the actually executed patched training
  notebook are missing locally.
- The targeted-v2 adapter and training metrics remain in Drive rather than in
  this local repository.
- There is no expert-authored golden set.
- Three provisional training rows retain reviewer-flagged academic errors:
  `case_uae_adab_default_arvid_fnbfetkhdc_03`,
  `case_uae_adab_default_arvid_fnbfetkhdc_06`, and
  `case_uae_adab_default_arvid_stz_chwha_04`.
- Each tuned condition used one training seed.
- Each final condition has one deterministic generated answer trajectory.
- The primary benchmark has ten scenario clusters and one external judge.
- No confidence interval or bootstrap analysis was run.
- Exact-silver v1 Complete-600 produced two factual hard-gate failures in the
  initial four-condition run and one in the later three-condition rerun.
- Targeted v2 added one religious-boundary breach and one academic-integrity
  breach in its controlled regression run.

These gaps do not invalidate the completed primary assignment comparison, but
they limit reproducibility, uncertainty estimates, and generalization claims.
