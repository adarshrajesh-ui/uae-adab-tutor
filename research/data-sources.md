# Data Sources — UAE Adab Academic Tutor Fine-Tune

Compiled 2026-07-09 from three background research workflows (bedrock + verification, 5-lane data sourcing, scout→3 novel-methods). Every link here was opened/verified by an agent unless flagged otherwise.

---

## TL;DR

- **There is no drop-in "adab tutor" dataset — and there shouldn't be.** If one existed, the behavior would fail the project's own litmus test (a prompt/dataset would already do it). The dataset is the deliverable.
- **The fine-tune runs on a dataset we *generate*** (synthetic multi-turn distillation), **seeded by real tutoring datasets, grounded in real adab texts + UAE gov docs, gated by two quality checks.** All of those ingredients are found, verified, and listed below.
- **Commercially safe to build on:** ConvoLearn (MIT), MathDial (CC-BY*), sunnah_ar_en (MIT), quran-qa (CC-BY), hh-rlhf schema (MIT), Qwen3 (Apache-2.0).
- **The one real gap:** Gulf/Emirati family-voice dialogue — mostly gated/paid; likely needs custom collection.

### Authentic-interaction supplement (2026-07-09)

A separate local supplement now contains five public-video transcripts selected for actual educator-child or classroom interaction rather than lectures: `research/scraped/authentic_cases/` (8,700 words; four English sources and one Arabic source). Four anonymized behavioral abstractions are stored in `research/authentic_case_seeds.json`. They cover validating a difficult question before explaining, gently clarifying a hesitant child's real question, connecting difficult curriculum questions to lived context, and warm structure in Arabic group learning. Two of the five videos proved narration-heavy and remain context-only rather than being mislabeled as dialogue. All five are standard-license and involve edited public footage or minors, so raw text remains local; no names or verbatim child dialogue may enter a published dataset.

---

## 1. Sources by role

### A. Base model & training tooling
| Asset | Link | License | Note |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 | Apache-2.0 | base model; 1.7B for fast iteration |
| Unsloth | https://github.com/unslothai/unsloth | Apache-2.0 | QLoRA; 4B fits one 24GB GPU |
| TRL (SFT + DPOTrainer) | https://huggingface.co/docs/trl/en/dpo_trainer | Apache-2.0 | exact preference JSONL schema |
| distilabel (Argilla) | https://github.com/argilla-io/distilabel | Apache-2.0 | generation + LLM-judge + dedup orchestration |

### B. Structural backbone — multi-turn tutoring datasets (clone the *structure*, re-author into adab)
| Dataset | Link | License | Size | Fit |
|---|---|---|---|---|
| **ConvoLearn** | https://huggingface.co/datasets/masharma/convolearn | **MIT** ✅ | 2,134 dialogues (~20 turns), Power-Dynamics + Cultural-Responsiveness labels | direct/adaptable — best find |
| **MathDial** | https://huggingface.co/datasets/eth-nlped/mathdial | CC-BY-4.0* ⚠️ | 2,861 dialogues, teacher-move taxonomy, non-answer-revealing scaffolding | direct/adaptable |
| SocraTeach / SocraticLM | https://github.com/Ljyustc/SocraticLM | CC-BY-NC 4.0 (data) | 35k multi-round + 22k single Socratic | adaptable (research) |
| Bridge | https://huggingface.co/datasets/rose-e-wang/bridge | CC-BY-NC 4.0 | 700 rows, novice+expert + error/strategy/**intention** | **ready-made DPO pairs** (research) |
| CIMA | https://github.com/kstats/CIMA | CC-BY 2.5 ✅ | ~2,970 tutor responses, action labels, multiple-valid-responses | adaptable |
| verify-then-generate | https://github.com/eth-lre/verify-then-generate | verify repo | 1,002 stepwise error-annotated chains | diagnose-before-correct |
| TSCC (Teacher-Student Chatroom) | https://aclanthology.org/2022.nlp4call-1.3.pdf | CC-BY-NC-SA 4.0 | 260 real ESL lessons, 41.4k turns | adaptable, **gated (Google Form)** |
| TalkMoves | https://github.com/SumnerLab/TalkMoves | CC-BY-NC-SA 4.0 | 567 K-12 transcripts, 10 accountable-talk moves | seed only |

\* MathDial license is reported inconsistently (HF card = CC-BY-4.0; paper repo = CC-BY-SA). **Verify on the card before relying on it.**

### C. Grounding & authenticity — adab tradition + UAE gov + Islamic text
| Source | Link | License | Use |
|---|---|---|---|
| al-Ghazālī, *Kitāb al-ʿIlm* (Iḥyāʾ Bk 1) | http://www.ghazali.org/rrs-ovr/ | Arabic PD; translations (Fons Vitae/Honerkamp 2015, Faris 1962) in copyright | RAG grounding; cite **print** editions, not the aggregator |
| al-Zarnūjī, *Taʿlīm al-Mutaʿallim* | https://archive.org/search?query=talim+al+mutaallim | Arabic PD (older translations PD) | **only source safe to quote verbatim** |
| UAE MoE Code of Conduct | https://u.ae/en/information-and-services/education/school-education-k-12/code-of-conduct-for-professionals-in-the-education-sector | MoE reserved | civic-floor grounding; PDF on assets.u.ae; cite, don't republish |
| UAE Moral Education / MSCS curriculum | https://u.ae/en/information-and-services/education/school-education-k-12/curricula-and-language-of-instruction- | MoE reserved | institutional-register + civic-floor anchor; re-verify per-grade links |
| sunnah_ar_en_dataset | https://huggingface.co/datasets/gurgutan/sunnah_ar_en_dataset | **MIT** ✅ | 50,762 hadith AR/EN — akhlaq grounding (filter to character chapters) |
| quran-question-answer-context | https://huggingface.co/datasets/nazimali/quran-question-answer-context | **CC-BY-4.0** ✅ | ~1,220 Quran Q/A/context — rich-register grounding |
| LCQA-Islamic | https://huggingface.co/datasets/Faiz28/LCQA-Islamic | CC-BY-NC-SA ⚠️ | 66,788 QA — grounding, **non-commercial** |

### D. Scenario-seed / phrasing bank (scrape → seeds only, never republished verbatim)
| Source | Link | License | Use |
|---|---|---|---|
| SeekersGuidance Answers + parenting articles | https://seekersguidance.org/answers/ | all rights reserved | **richest on-thesis seed corpus** (~20k scholar Q&A; tarbiya/taʾdīb) |
| Yaqeen Institute — parenting hub | https://yaqeeninstitute.org/topic/parenting | copyrighted (auto transcripts) | seed/style; down-weight transcript accuracy |
| YouTube — Our Muslim Homeschool | https://www.youtube.com/channel/UCasrAhic7AwRbjAcIde5XiA/videos | YouTube ToS | family-register phrasing; strip PII |
| YouTube — "Tarbiyah & Adab" playlist | https://www.youtube.com/playlist?list=PL3CctSqNJJXhDWzLOMDUVO2pBfme3oN-O | YouTube ToS | high topical precision |
| Quranic Tarbiyah curriculum (Adab & Akhlaq) | https://quranictarbiyah.com/ | verify terms | structured adab-lesson scaffolding |

> **Pipeline already exists in this repo:** `scrape_transcripts.py`, `scrape_articles.py`, `transcript_utils.py` (yt-dlp for discovery, youtube-transcript-api for text — both installed & working). Plumbing is done.

### E. Civic-floor / safety preference structure (reuse the *schema*, not the content)
| Source | Link | License | Use |
|---|---|---|---|
| Anthropic/hh-rlhf | https://huggingface.co/datasets/Anthropic/hh-rlhf | **MIT** ✅ | chosen/rejected schema for civic-floor DPO pairs |
| PKU-SafeRLHF | https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF | CC-BY-NC ⚠️ | per-response safety flags + harm taxonomy (non-commercial) |

### F. Gulf/Emirati dialect — **the gap**
| Source | Link | License | Note |
|---|---|---|---|
| ArSyra | https://huggingface.co/datasets/ArSyra/arsyra-complete | preview CC-BY-NC-SA; full **paid** ($29+) | 127k records incl. Gulf; only 50-row preview free |

> Emirati family-voice dialogue is thin and mostly gated/paid. **Plan to collect this yourself** for the rich/family register.

### G. Method references (technique, not data)
Self-Instruct (arXiv 2212.10560) · Evol-Instruct/WizardLM (2304.12244) · Crescendo + PyRIT (2404.01833) · DPO (2305.18290) · ConsistentChat skeleton-then-realize (2506.03558) · KMP-Bench "From Solver to Tutor" (2603.02775) · Constitutional AI (2212.08073) · Fine-Grained RLHF (2306.01693). Justification: **SafeTutors** (arXiv 2603.17373 — single-turn 17.7% → multi-turn 77.8% pedagogical failure).

---

## 2. Licensing at a glance
- **Commercial / shippable (UAE-gov safe):** ConvoLearn (MIT), MathDial (CC-BY*), CIMA (CC-BY 2.5), sunnah_ar_en (MIT), quran-qa (CC-BY), hh-rlhf (MIT), Qwen3 (Apache-2.0), Unsloth/TRL/distilabel (Apache-2.0).
- **Research / demo only (NON-commercial):** SocraTeach, Bridge, TSCC, TalkMoves, LCQA-Islamic, PKU-SafeRLHF.
- **Paid/gated:** ArSyra (full set).
- **Ground/cite only, never republish:** al-Ghazālī/al-Zarnūjī translations, MoE docs, SeekersGuidance, Yaqeen, YouTube.
- Net exposure is **low** because the shipped dataset is synthetic/re-authored — but do a license audit before any commercial release and don't republish source text.

---

## 3. How each source feeds the dataset
1. **Skeletons** (Move-Grammar Compiler) — move taxonomy informed by ConvoLearn + MathDial + TalkMoves.
2. **Realizer** (frontier teacher model) — writes each turn, *grounded* on al-Ghazālī/al-Zarnūjī + MoE Code of Conduct + hadith/quran + scraped SeekersGuidance seeds.
3. **SFT set** (core) — ~1.5–2.5k gated multi-turn lessons, both registers, response-only loss.
4. **DPO pairs** — Bridge novice-vs-expert (ready-made) + gate rejects + register/floor twins + base-model failures.
5. **Quality gate** — the "delete-the-sentence" test (kills value-stickers) + a cross-family LLM judge (rubric reuses `prompt_test.py` failure tags). SafeTutors defines the failure taxonomy.
6. **Held-out eval** — adversarial multi-turn, sliced by pressure level and by register.

---

## 4. The generation engine (novel design, from the scout workflow — all 3 deep-dives "viable")
**`Move-Grammar Compiler → Register-Fork Twins → Muhtasib turn-critic`**
- **Move-Grammar Compiler** — symbolic FSM emits a JSON lesson skeleton (per-turn required move + pressure P 0–5 + register) *before* text, so durability coverage is guaranteed, not hoped for. (Precedent: ConsistentChat, KMP-Bench.) *Fix: make P non-monotonic.*
- **Register-Fork Twins** — realize each skeleton twice (light/institutional vs rich/family) from an identical skeleton → both-register SFT + minimal-pair DPO. Produces the two-axis eval (register-hold × floor-hold). *Depends on the compiler.*
- **Muhtasib turn-critic** — frontier critic scores every turn on a 6-check adab rubric, keeps passers for SFT, emits (original→minimal-rewrite) turn-level DPO pairs. *Risk: eval circularity — don't reuse the same rubric+model as judge; never use the small target model as critic.*
- **Feeders:** base-model failure farming (model-matched negatives), confused-student engine (authentic naive errors), subject×virtue×pressure coverage matrix (generalization), Arabizi/Gulf augmentation (input robustness).

---

## 5. Gaps, flags & dead ends
- 🚩 **`moraleducation.ae` is domain-hijacked to a gambling site — DO NOT USE.** Substitute the official MOE e-reader + archived teacher-guide PDFs.
- ⚠️ **Domain skew:** nearly all open tutoring data is *math*. Must augment to other subjects or adab only holds in math.
- ⚠️ **MathDial license** inconsistent (CC-BY vs CC-BY-SA) — verify.
- ⚠️ **Gulf/Emirati dialogue gap** — collect custom for family register.
- ⚠️ **Authenticity/caricature** — synthetic adab drifts to stereotype without in-community review (budget the Tabah/specialist pass).
- ⚠️ **Judge validity** — validate cross-family judge κ on the adab target *before* scaling data spend; if annotators can't agree on the deletion test, the spec is too soft to train. (#1 project risk.)

---

## 6. Provenance
Compiled from workflow runs `wf_ae060abf` (bedrock), `wf_a7bcbc72` (verification — all 5 bedrock items confirmed real), `wf_48a0a23b` (5-lane data sourcing), `wf_cfe93dc7` (scout→3 novel methods). ~26 agents, adversarially cross-checked. Treat single-agent claims as leads; re-verify a link before it becomes load-bearing.
