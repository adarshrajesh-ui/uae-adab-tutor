#!/usr/bin/env python3
"""Assemble the frozen 120-grounded/480-revised experimental silver release.

Unlike ``assemble_exact_v2_release.py``, this deadline snapshot permits only
explicitly disclosed provisional rows.  It never upgrades those rows to an
accepted-review status.  Every included row is bound to a per-row limitation
entry, all lineage groups remain in one split, and the three frozen evaluation
suites remain external.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assemble_exact_v2_release import (  # noqa: E402
    REVISED_EXPRESSION_COUNTS,
    load_revised_components,
)
from assemble_silver_splits import (  # noqa: E402
    DEFAULT_EVALS,
    DEFAULT_HELDOUTS,
    canonical_hash,
    choose_groups,
    distribution,
    eval_leakage,
    file_sha256,
    group_key,
    read_json,
    read_jsonl,
    row_id,
    with_split,
    write_jsonl,
)


DEFAULT_GROUNDED_DIR = ROOT / "data/uae_adab/v3/final_600/grounded_120_silver_v1"
DEFAULT_REVISED_BASE = (
    ROOT
    / "data/uae_adab/v2/single_register/recovery_v1/final_closure_repair/"
    "rebalanced_v6/accepted_505_split/accepted_505_all.jsonl"
)
DEFAULT_GAP_DIR = ROOT / "data/uae_adab/v3/final_600/revised_gap_95/exact95_silver_freeze"
DEFAULT_COMPILER = ROOT / "data/uae_adab/v3/final_600/revised_gap_95/compiler_final_revised_408_72.json"
DEFAULT_WORKLIST = ROOT / "data/uae_adab/v3/final_600/revised_gap_95/worklist_missing_95.jsonl"
DEFAULT_OUTPUT = ROOT / "data/uae_adab/v3/final_600/exact_silver_release_v1"

SYSTEM = "<uae_adab_tutor>default</uae_adab_tutor>"
GROUND_EXPRESSION_COUNTS = {"shared_implicit": 102, "explicit_sparse": 18}
GROUND_FAMILY_COUNTS = {
    "MathDial": 64,
    "ConvoLearn": 35,
    "Arabic YouTube academic lesson": 21,
}
COMPLETE_EXPRESSION_COUNTS = {"shared_implicit": 510, "explicit_sparse": 90}


class SilverAssemblyError(ValueError):
    """The frozen snapshot violated a release invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SilverAssemblyError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def counter(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def canonical_tier(row: dict[str, Any]) -> str:
    freeze = row.get("silver_freeze")
    if isinstance(freeze, dict) and isinstance(freeze.get("tier"), str):
        return freeze["tier"]
    quality = row.get("quality")
    if isinstance(quality, dict):
        tier = quality.get("silver_tier") or quality.get("review_tier")
        if isinstance(tier, str) and tier:
            return tier
    status = row.get("status")
    mapping = {
        "accepted_expedited_silver": "reviewed_accepted_expedited",
        "accepted_unanimous_review": "reviewed_accepted_dual",
        "accepted_grounded_v3": "reviewed_accepted_grounded_v3",
        "candidate_unreviewed": "unreviewed_structural_preflight",
    }
    return mapping.get(str(status), str(status or "unknown"))


def validate_messages(row: dict[str, Any]) -> None:
    identifier = row_id(row)
    messages = row.get("messages")
    require(isinstance(messages, list) and len(messages) >= 3, f"{identifier} lacks messages")
    require(
        messages[0] == {"role": "system", "content": SYSTEM},
        f"{identifier} has wrong system message",
    )
    expected = "user"
    for index, message in enumerate(messages[1:], 1):
        require(
            isinstance(message, dict)
            and set(message) == {"role", "content"}
            and message.get("role") == expected
            and isinstance(message.get("content"), str)
            and bool(message["content"].strip()),
            f"{identifier} message {index} is malformed",
        )
        expected = "assistant" if expected == "user" else "user"
    require(messages[-1]["role"] == "assistant", f"{identifier} does not end with tutor")
    require(
        canonical_hash(messages) == row.get("conversation_sha256"),
        f"{identifier} conversation hash mismatch",
    )


def validate_unsplit_rows(rows: list[dict[str, Any]], role: str) -> None:
    for row in rows:
        require(row.get("corpus_role") == role, f"{row_id(row)} has wrong corpus role")
        require("split" not in row and "split_group_sha256" not in row, f"{row_id(row)} is pre-split")
        validate_messages(row)


def load_status_ledger(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = row.get("source_id") or row.get("compiler_assignment_id")
        require(isinstance(source_id, str) and source_id, f"{path} has invalid source ID")
        require(source_id not in output, f"{path} duplicates {source_id}")
        output[source_id] = row
    return output


def validate_grounded(
    rows: list[dict[str, Any]], status_ledger: dict[str, dict[str, Any]]
) -> None:
    require(len(rows) == 120, f"grounded snapshot must contain 120 rows; got {len(rows)}")
    validate_unsplit_rows(rows, "grounded")
    ids = {row_id(row) for row in rows}
    require(ids == set(status_ledger), "grounded status ledger does not exactly cover 120 rows")
    require(
        counter(rows, "expression_class") == GROUND_EXPRESSION_COUNTS,
        f"grounded expression balance differs: {counter(rows, 'expression_class')}",
    )
    families = Counter(row.get("provenance", {}).get("source_family") for row in rows)
    require(
        dict(sorted(families.items())) == GROUND_FAMILY_COUNTS,
        f"grounded source-family balance differs: {dict(families)}",
    )
    for row in rows:
        identifier = row_id(row)
        entry = status_ledger[identifier]
        require(
            entry.get("conversation_sha256") == row.get("conversation_sha256"),
            f"{identifier} status ledger hash mismatch",
        )
        require(
            entry.get("trainable_in_silver") is True,
            f"{identifier} is not explicitly trainable in the status ledger",
        )
        provenance = row.get("provenance")
        require(isinstance(provenance, dict), f"{identifier} lacks provenance")
        require(
            isinstance(provenance.get("source_group_id"), str)
            and bool(provenance["source_group_id"]),
            f"{identifier} lacks source_group_id",
        )


def validate_revised(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == 480, f"revised snapshot must contain 480 rows; got {len(rows)}")
    validate_unsplit_rows(rows, "revised")
    require(
        counter(rows, "expression_class") == REVISED_EXPRESSION_COUNTS,
        f"revised expression balance differs: {counter(rows, 'expression_class')}",
    )
    for row in rows:
        freeze = row.get("silver_freeze")
        if isinstance(freeze, dict):
            require(
                freeze.get("trainable_in_silver") is True,
                f"{row_id(row)} gap row is not explicitly trainable in silver",
            )
        else:
            require(
                row.get("status") in {
                    "accepted_expedited_silver",
                    "accepted_unanimous_review",
                },
                f"{row_id(row)} base revised row is neither accepted nor silver-disclosed",
            )


def source_family_key(row: dict[str, Any]) -> str | None:
    if row.get("corpus_role") != "grounded":
        return None
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("source_family_key") or provenance.get("source_group_id")
    return value if isinstance(value, str) and value else None


def assert_split_isolation(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> dict[str, int]:
    group_overlap = {group_key(row) for row in train} & {
        group_key(row) for row in validation
    }
    id_overlap = {row_id(row) for row in train} & {row_id(row) for row in validation}
    source_overlap = {
        key for row in train if (key := source_family_key(row)) is not None
    } & {
        key for row in validation if (key := source_family_key(row)) is not None
    }
    require(not group_overlap, f"train/validation lineage overlap: {sorted(group_overlap)}")
    require(not id_overlap, f"train/validation ID overlap: {sorted(id_overlap)}")
    require(not source_overlap, f"train/validation source-family overlap: {sorted(source_overlap)}")
    return {
        "train_validation_group_overlap": 0,
        "train_validation_id_overlap": 0,
        "train_validation_source_family_overlap": 0,
    }


def portable_artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"file": path.name, "sha256": file_sha256(path)}
    if rows is not None:
        result["rows"] = rows
    return result


def limitations_for(row: dict[str, Any]) -> list[str]:
    freeze = row.get("silver_freeze")
    if isinstance(freeze, dict) and isinstance(freeze.get("limitation"), str):
        return [freeze["limitation"]]
    quality = row.get("quality")
    if isinstance(quality, dict):
        values = quality.get("limitations")
        if isinstance(values, list):
            return [str(value) for value in values]
        value = quality.get("limitation")
        if isinstance(value, str) and value:
            return [value]
    return []


def build_limitation_ledger(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "compiler_assignment_id": row_id(row),
            "conversation_sha256": row["conversation_sha256"],
            "corpus_role": row["corpus_role"],
            "expression_class": row["expression_class"],
            "split": row["split"],
            "status": row.get("status"),
            "tier": canonical_tier(row),
            "limitations": limitations_for(row),
            "trainable_in_this_experimental_silver_release": True,
            "publication_grade": False,
        }
        for row in sorted(rows, key=row_id)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grounded",
        type=Path,
        default=DEFAULT_GROUNDED_DIR / "grounded_120_normalized.jsonl",
    )
    parser.add_argument(
        "--grounded-status-ledger",
        type=Path,
        default=DEFAULT_GROUNDED_DIR / "case_status_ledger_120.jsonl",
    )
    parser.add_argument("--revised-base", type=Path, default=DEFAULT_REVISED_BASE)
    parser.add_argument(
        "--revised-gap", type=Path, default=DEFAULT_GAP_DIR / "silver_gap_exact95.jsonl"
    )
    parser.add_argument(
        "--revised-gap-ledger",
        type=Path,
        default=DEFAULT_GAP_DIR / "limitation_ledger_95.jsonl",
    )
    parser.add_argument("--revised-gap-worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--eval-scenarios", type=Path, action="append", default=None)
    parser.add_argument("--heldout-groups", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", default="uae-adab-exact-silver-600-v1")
    parser.add_argument("--trials", type=int, default=5000)
    return parser.parse_args()


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    expected_inputs = (
        args.grounded,
        args.grounded_status_ledger,
        args.revised_base,
        args.revised_gap,
        args.revised_gap_ledger,
        args.revised_gap_worklist,
        args.compiler,
    )
    missing = [str(path) for path in expected_inputs if not path.is_file()]
    require(not missing, "missing required input(s): " + ", ".join(missing))
    require(not args.output_dir.exists(), f"output already exists: {args.output_dir}")
    require(args.trials > 0, "trials must be positive")

    grounded = sorted(read_jsonl(args.grounded), key=row_id)
    grounded_status = load_status_ledger(args.grounded_status_ledger)
    validate_grounded(grounded, grounded_status)
    revised, revised_report = load_revised_components(
        base_path=args.revised_base,
        gap_path=args.revised_gap,
        gap_worklist_path=args.revised_gap_worklist,
        compiler=read_json(args.compiler),
    )
    revised = sorted(revised, key=row_id)
    validate_revised(revised)
    all_rows = grounded + revised
    require(len(all_rows) == 600, "combined corpus is not exactly 600 rows")
    identifiers = [row_id(row) for row in all_rows]
    hashes = [row.get("conversation_sha256") for row in all_rows]
    require(len(identifiers) == len(set(identifiers)), "duplicate compiler assignment IDs")
    require(len(hashes) == len(set(hashes)), "duplicate conversation hashes")
    require(
        counter(all_rows, "expression_class") == COMPLETE_EXPRESSION_COUNTS,
        f"complete expression balance differs: {counter(all_rows, 'expression_class')}",
    )

    heldout_group_paths = list(args.heldout_groups or DEFAULT_HELDOUTS)
    reserved_groups = {
        group
        for path in heldout_group_paths
        for group in read_json(path).get("source_group_ids", [])
    }
    overlaps = sorted(
        {
            row.get("provenance", {}).get("source_group_id")
            for row in grounded
            if row.get("provenance", {}).get("source_group_id") in reserved_groups
        }
    )
    require(not overlaps, f"grounded rows overlap reserved source groups: {overlaps}")

    eval_paths = list(args.eval_scenarios or DEFAULT_EVALS)
    leakage = eval_leakage(all_rows, eval_paths)
    require(
        not leakage["exact_overlap_rows"] and not leakage["twelve_gram_overlap_rows"],
        f"frozen evaluation leakage detected: {leakage}",
    )

    grounded_validation_groups = choose_groups(
        grounded,
        12,
        seed=f"{args.seed}:grounded-validation",
        trials=args.trials,
    )
    revised_validation_groups = choose_groups(
        revised,
        48,
        seed=f"{args.seed}:revised-validation",
        trials=args.trials,
    )
    grounded_train = [
        with_split(row, "train")
        for row in grounded
        if group_key(row) not in grounded_validation_groups
    ]
    grounded_validation = [
        with_split(row, "validation")
        for row in grounded
        if group_key(row) in grounded_validation_groups
    ]
    revised_train = [
        with_split(row, "train")
        for row in revised
        if group_key(row) not in revised_validation_groups
    ]
    revised_validation = [
        with_split(row, "validation")
        for row in revised
        if group_key(row) in revised_validation_groups
    ]
    complete_train = grounded_train + revised_train
    complete_validation = grounded_validation + revised_validation
    grounded_all = grounded_train + grounded_validation
    complete_all = complete_train + complete_validation
    counts = {
        "grounded": {
            "all": len(grounded_all),
            "train": len(grounded_train),
            "validation": len(grounded_validation),
        },
        "revised": {
            "all": len(revised),
            "train": len(revised_train),
            "validation": len(revised_validation),
        },
        "complete": {
            "all": len(complete_all),
            "train": len(complete_train),
            "validation": len(complete_validation),
        },
    }
    require(
        counts
        == {
            "grounded": {"all": 120, "train": 108, "validation": 12},
            "revised": {"all": 480, "train": 432, "validation": 48},
            "complete": {"all": 600, "train": 540, "validation": 60},
        },
        f"split counts differ: {counts}",
    )
    grouping = assert_split_isolation(complete_train, complete_validation)
    require(
        sum(row["corpus_role"] == "grounded" for row in complete_train) == 108,
        "complete train is not exactly 20% grounded",
    )
    require(
        sum(row["corpus_role"] == "grounded" for row in complete_validation) == 12,
        "complete validation is not exactly 20% grounded",
    )
    grounded_index = {
        row_id(row): (row["conversation_sha256"], row["split"], row["split_group_sha256"])
        for row in grounded_all
    }
    complete_grounded_index = {
        row_id(row): (row["conversation_sha256"], row["split"], row["split_group_sha256"])
        for row in complete_all
        if row["corpus_role"] == "grounded"
    }
    require(
        grounded_index == complete_grounded_index,
        "grounded rows changed inside complete corpus",
    )

    paths = {
        "grounded120_all": args.output_dir / "grounded_120_all.jsonl",
        "grounded120_train": args.output_dir / "grounded_120_train.jsonl",
        "grounded120_validation": args.output_dir / "grounded_120_validation.jsonl",
        "complete600_all": args.output_dir / "complete_600_all.jsonl",
        "complete600_train": args.output_dir / "complete_600_train.jsonl",
        "complete600_validation": args.output_dir / "complete_600_validation.jsonl",
        "limitation_ledger": args.output_dir / "release_limitation_ledger_600.jsonl",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(paths["grounded120_all"], grounded_all)
    write_jsonl(paths["grounded120_train"], grounded_train)
    write_jsonl(paths["grounded120_validation"], grounded_validation)
    write_jsonl(paths["complete600_all"], complete_all)
    write_jsonl(paths["complete600_train"], complete_train)
    write_jsonl(paths["complete600_validation"], complete_validation)
    limitation_rows = build_limitation_ledger(complete_all)
    write_jsonl(paths["limitation_ledger"], limitation_rows)

    status_counts = dict(sorted(Counter(row.get("status") for row in complete_all).items()))
    tier_counts = dict(sorted(Counter(canonical_tier(row) for row in complete_all).items()))
    split_tier_counts = {
        split: dict(
            sorted(
                Counter(canonical_tier(row) for row in complete_all if row["split"] == split).items()
            )
        )
        for split in ("train", "validation")
    }
    artifacts = {
        name: portable_artifact(path, 600 if name == "limitation_ledger" else len(read_jsonl(path)))
        for name, path in paths.items()
    }
    manifest = {
        "schema_version": "uae_adab_exact_silver_600_release.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_authority": False,
        "training_use": "time-boxed experimental silver; not a publication-grade gold corpus",
        "counts": counts,
        "composition": {
            "source_grounded_rows": 120,
            "synthetic_revised_rows": 480,
            "source_grounded_fraction": 0.20,
            "train_source_grounded_fraction": 0.20,
            "validation_source_grounded_fraction": 0.20,
            "expression_class": counter(complete_all, "expression_class"),
            "grounded_source_family": dict(
                sorted(
                    Counter(
                        row.get("provenance", {}).get("source_family")
                        for row in grounded_all
                    ).items()
                )
            ),
        },
        "status_counts": status_counts,
        "tier_counts": tier_counts,
        "split_tier_counts": split_tier_counts,
        "grouping": {
            **grouping,
            "grounded_rule": "source record; all segments from one Arabic video stay together",
            "revised_rule": "subject + scenario_family + learning_objective",
            "seed": args.seed,
        },
        "subset_audit": {
            "same_ids_hashes_splits_and_groups": True,
            "grounded_rows_in_complete": 120,
        },
        "heldout": {
            "frozen_evaluations_remain_external_test_data": True,
            "reserved_source_groups": len(reserved_groups),
            "training_source_group_overlap": 0,
            "evaluation_leakage": leakage,
        },
        "input_artifacts": {
            "grounded": portable_artifact(args.grounded, 120),
            "grounded_status_ledger": portable_artifact(args.grounded_status_ledger, 120),
            "revised_base": portable_artifact(args.revised_base, 505),
            "revised_gap": portable_artifact(args.revised_gap, 95),
            "revised_gap_ledger": portable_artifact(args.revised_gap_ledger, 95),
            "revised_gap_worklist": portable_artifact(args.revised_gap_worklist, 95),
            "compiler": portable_artifact(args.compiler),
        },
        "revised_component_report": revised_report,
        "artifacts": artifacts,
        "limitations": [
            "This release is exactly 600 trainable rows but is an experimental silver snapshot, not 600 reviewed gold rows.",
            "Grounded v3 has 108 fresh dual-accepted rows, 11 fresh reviewer-disputed rows, and one deterministic-valid writer-unreviewed row.",
            "The revised 95-row gap has 58 reviewed acceptances, 36 structurally preflighted but unreviewed rows, and one explicitly marked prior-review-rejected fallback.",
            "The immutable revised base uses its documented expedited/selective review policy rather than universal dual review.",
            "No domain-expert human golden set was available; frozen adversarial and source-transfer evaluations remain the unbiased test sets.",
            "YouTube-derived rows rely on the project owner's permission attestation and nonverbatim translated incorporation.",
        ],
    }
    manifest_path = args.output_dir / "exact_silver_release_manifest.json"
    write_json(manifest_path, manifest)
    note_path = args.output_dir / "LIMITATIONS.md"
    note_path.write_text(
        "# Exact 600 experimental silver snapshot\n\n"
        "This release has 600 rows and an exact 540/60 group-safe split. Exactly 120 rows "
        "(20%) visibly incorporate registered source substance; 480 are synthetic revised "
        "behavior cases. It is deliberately labeled silver. Grounded v3 contains 108 fresh "
        "dual-accepted cases, 11 reviewer-disputed cases, and one deterministic-valid but "
        "writer-unreviewed reconstruction. The revised gap contains 58 reviewed cases, 36 "
        "structural-preflight-only cases, and one prior-review-rejected fallback. See "
        "`release_limitation_ledger_600.jsonl` for the exact row-level status.\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def main() -> int:
    try:
        report = assemble(parse_args())
    except (SilverAssemblyError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "tier_counts": report["tier_counts"],
                "manifest": report["manifest_path"],
                "manifest_sha256": report["manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
