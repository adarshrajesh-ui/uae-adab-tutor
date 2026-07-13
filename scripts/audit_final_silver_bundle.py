#!/usr/bin/env python3
"""Independent read-only audit of the expedited silver release bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data/uae_adab/v2/single_register/silver_v0/release_inputs"
RELEASE_DIR = ROOT / "data/uae_adab/v2/single_register/silver_v0/release"
WORKLIST = ROOT / "data/uae_adab/v2/single_register_compiler_worklist.json"
EXCLUSIONS = ROOT / "research/grounded_v2/source_fidelity/grounded_release_exclusions.json"
HELDOUTS = [
    ROOT / "research/grounded_v2/normalized/mathdial/final_single_register_v2/heldout_source_groups.json",
    ROOT / "research/grounded_v2/normalized/convolearn/final_single_register_v2/heldout_source_groups.json",
]
EVALS = [
    ROOT / "evals/uae_adab/scenarios.json",
    ROOT / "evals/uae_adab/authentic_heldout_scenarios.json",
    ROOT / "evals/uae_adab/source_transfer_scenarios_v0.json",
]
SYSTEM = "<uae_adab_tutor>default</uae_adab_tutor>"
JSONL_INPUTS = {
    "grounded_accepted": INPUT_DIR / "grounded_accepted.jsonl",
    "revised_accepted": INPUT_DIR / "revised_accepted.jsonl",
    "grounded_dropped": INPUT_DIR / "grounded_dropped.jsonl",
    "revised_dropped": INPUT_DIR / "revised_dropped.jsonl",
    "drop_ledger": INPUT_DIR / "drop_ledger.jsonl",
}
RELEASE_JSONL = {
    name: RELEASE_DIR / f"{name}.jsonl"
    for name in (
        "grounded_all", "grounded_train", "grounded_validation",
        "mixed_all", "mixed_train", "mixed_validation",
        "balanced20_all", "balanced20_train", "balanced20_validation",
    )
}
TOKEN_REPORTS = {
    "grounded_all": RELEASE_DIR / "grounded_token_lengths.json",
    "mixed_all": RELEASE_DIR / "mixed_token_lengths.json",
    "balanced20_all": RELEASE_DIR / "balanced20_token_lengths.json",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ASSEMBLE = load_module("bundle_assemble", ROOT / "scripts/assemble_silver_splits.py")
TOKENS = load_module("bundle_tokens", ROOT / "scripts/analyze_qwen_token_lengths.py")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["compiler_assignment_id"], row["conversation_sha256"]


def row_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {row_key(row): row for row in rows}


def token_report(rows: list[dict[str, Any]], input_path: Path, tokenizer: Any) -> dict[str, Any]:
    measurements = []
    for row in rows:
        token_ids = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        assistant_tokens = sum(
            len(tokenizer.encode(message["content"], add_special_tokens=False))
            for message in row["messages"]
            if message["role"] == "assistant"
        )
        measurements.append(
            {
                "id": row["id"],
                "split": row["split"],
                "subject": TOKENS.metadata_value(row, "subject"),
                "register": TOKENS.metadata_value(row, "register"),
                "tokens": len(token_ids),
                "assistant_content_tokens": assistant_tokens,
            }
        )
    full = [item["tokens"] for item in measurements]
    assistant = [item["assistant_content_tokens"] for item in measurements]
    thresholds = [2048, 4096]
    return {
        "input": str(input_path.relative_to(ROOT)),
        "tokenizer": "Qwen/Qwen3-4B-Instruct-2507",
        "rows": len(rows),
        "full_conversation_tokens": TOKENS.summarize(full),
        "assistant_content_tokens": TOKENS.summarize(assistant),
        "thresholds": {
            str(threshold): {
                "rows_exceeding": sum(value > threshold for value in full),
                "percent_exceeding": round(100 * sum(value > threshold for value in full) / len(full), 2),
                "tokens_truncated_if_right_truncated": sum(max(0, value - threshold) for value in full),
            }
            for threshold in thresholds
        },
        "by_split": {
            split: TOKENS.summarize([item["tokens"] for item in measurements if item["split"] == split])
            for split in sorted({item["split"] for item in measurements})
        },
        "by_register": {
            register: TOKENS.summarize([item["tokens"] for item in measurements if item["register"] == register])
            for register in sorted({item["register"] for item in measurements})
        },
        "subject_counts": dict(sorted(Counter(item["subject"] for item in measurements).items())),
        "longest_records": sorted(measurements, key=lambda item: (-item["tokens"], item["id"]))[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RELEASE_DIR / "final_bundle_audit.json")
    parser.add_argument("--skip-token-recompute", action="store_true")
    args = parser.parse_args()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def fail(code: str, detail: Any) -> None:
        errors.append({"code": code, "detail": detail})

    required = [
        *JSONL_INPUTS.values(), *RELEASE_JSONL.values(), *TOKEN_REPORTS.values(),
        INPUT_DIR / "release_input_report.json", INPUT_DIR / "revised_release_gate.json",
        RELEASE_DIR / "split_report.json", RELEASE_DIR / "LIMITATIONS.md",
        WORKLIST, EXCLUSIONS, *HELDOUTS, *EVALS,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        fail("missing_bundle_files", missing)

    inputs = {name: read_jsonl(path) for name, path in JSONL_INPUTS.items() if path.exists()}
    release = {name: read_jsonl(path) for name, path in RELEASE_JSONL.items() if path.exists()}
    input_report = read_json(INPUT_DIR / "release_input_report.json")
    split_report = read_json(RELEASE_DIR / "split_report.json")
    revised_gate = read_json(INPUT_DIR / "revised_release_gate.json")
    worklist = read_json(WORKLIST)
    assignment_rows = worklist["assignments"]
    assignments = {row["assignment_id"]: row for row in assignment_rows}
    reserved_groups = {
        group for path in HELDOUTS for group in read_json(path).get("source_group_ids", [])
    }

    # Reports must bind every materialized file by exact row count and SHA-256.
    report_hash_errors = []
    for name in ("grounded_accepted", "revised_accepted", "grounded_dropped", "revised_dropped", "drop_ledger"):
        expected = input_report["outputs"][name]
        path = JSONL_INPUTS[name]
        if expected["rows"] != len(inputs[name]) or expected["sha256"] != file_hash(path):
            report_hash_errors.append(name)
    for name, path in RELEASE_JSONL.items():
        expected = split_report["artifacts"].get(name)
        if expected is None or expected["rows"] != len(release[name]) or expected["sha256"] != file_hash(path):
            report_hash_errors.append(name)
    if report_hash_errors:
        fail("report_artifact_hash_or_count_mismatch", report_hash_errors)
    if input_report.get("schema_version") != "uae_adab_expedited_silver_release_inputs.v1" or input_report.get("ok") is not True:
        fail("release_input_report_schema_or_status", input_report.get("schema_version"))
    if split_report.get("schema_version") != "uae_adab_expedited_silver_split.v0":
        fail("split_report_schema", split_report.get("schema_version"))
    if revised_gate.get("decision") != "authorize" or revised_gate.get("proposed", {}).get("sha256") != file_hash(JSONL_INPUTS["revised_accepted"]):
        fail("revised_release_gate_binding", revised_gate.get("decision"))

    grounded = inputs["grounded_accepted"]
    revised = inputs["revised_accepted"]
    accepted = grounded + revised
    dropped = inputs["grounded_dropped"] + inputs["revised_dropped"]
    accepted_ids = {row["compiler_assignment_id"] for row in accepted}
    dropped_ids = {row["compiler_assignment_id"] for row in dropped}
    universe_ids = set(assignments)
    if accepted_ids & dropped_ids or accepted_ids | dropped_ids != universe_ids:
        fail(
            "accepted_dropped_partition",
            {
                "overlap": sorted(accepted_ids & dropped_ids),
                "missing": sorted(universe_ids - (accepted_ids | dropped_ids)),
                "extra": sorted((accepted_ids | dropped_ids) - universe_ids),
            },
        )
    if len(accepted) != 258 or len(dropped) != 342 or len(universe_ids) != 600:
        fail("bundle_universe_counts", {"accepted": len(accepted), "dropped": len(dropped), "universe": len(universe_ids)})
    if {canonical_hash(row) for row in inputs["drop_ledger"]} != {canonical_hash(row) for row in dropped}:
        fail("drop_ledger_not_exact_union", None)
    if len(inputs["drop_ledger"]) != len({row["compiler_assignment_id"] for row in inputs["drop_ledger"]}):
        fail("drop_ledger_duplicate_ids", None)

    try:
        ASSEMBLE.validate_rows(accepted, reserved_groups)
        ASSEMBLE.validate_rows(release["mixed_all"], reserved_groups)
    except Exception as exc:
        fail("accepted_schema_or_source_group_contract", f"{type(exc).__name__}: {exc}")
    wrong_system = [row["compiler_assignment_id"] for row in release["mixed_all"] if row["messages"][0] != {"role": "system", "content": SYSTEM}]
    if wrong_system:
        fail("wrong_system_message", wrong_system)

    # Forced exact source/conversation exclusions must stay absent and auditable.
    exclusion_doc = read_json(EXCLUSIONS)
    payload = exclusion_doc["audit_payload"]
    if exclusion_doc.get("audit_sha256") != canonical_hash(payload):
        fail("exclusion_envelope_hash", None)
    excluded_versions = {
        (row["source_id"], row["conversation_sha256"]) for row in payload["exclusions"]
    }
    observed_grounded_versions = {
        (row.get("provenance", {}).get("source_id"), row["conversation_sha256"]) for row in grounded
    }
    if excluded_versions & observed_grounded_versions:
        fail("forced_excluded_grounded_version_present", sorted(excluded_versions & observed_grounded_versions))
    excluded_source_ids = {source for source, _ in excluded_versions}
    if not excluded_source_ids <= dropped_ids:
        fail("forced_exclusions_missing_from_drop_partition", sorted(excluded_source_ids - dropped_ids))
    grounded_source_ids = [row.get("provenance", {}).get("source_id") for row in grounded]
    if len(grounded_source_ids) != len(set(grounded_source_ids)) or set(grounded_source_ids) != {
        row["compiler_assignment_id"] for row in grounded
    }:
        fail("grounded_source_assignment_identity", None)

    # Every release row must be an exact accepted row plus deterministic split metadata.
    accepted_by_key = row_map(accepted)
    membership_errors = []
    for name, rows in release.items():
        expected_split = "train" if name.endswith("_train") else ("validation" if name.endswith("_validation") else None)
        for row in rows:
            original = accepted_by_key.get(row_key(row))
            if original is None:
                membership_errors.append({"artifact": name, "id": row.get("compiler_assignment_id"), "reason": "not accepted input"})
                continue
            split = row.get("split")
            if expected_split is not None and split != expected_split:
                membership_errors.append({"artifact": name, "id": row["compiler_assignment_id"], "reason": "wrong split"})
            if row != ASSEMBLE.with_split(original, split):
                membership_errors.append({"artifact": name, "id": row["compiler_assignment_id"], "reason": "content differs from accepted+split"})
    if membership_errors:
        fail("release_membership_or_content", membership_errors[:100])

    expected_sets = {
        "grounded_all": {row_key(row) for row in grounded},
        "mixed_all": {row_key(row) for row in accepted},
    }
    for name, expected in expected_sets.items():
        if {row_key(row) for row in release[name]} != expected:
            fail("all_artifact_membership", name)
    for prefix in ("grounded", "mixed", "balanced20"):
        all_keys = {row_key(row) for row in release[f"{prefix}_all"]}
        train_keys = {row_key(row) for row in release[f"{prefix}_train"]}
        val_keys = {row_key(row) for row in release[f"{prefix}_validation"]}
        if train_keys & val_keys or train_keys | val_keys != all_keys:
            fail("split_row_partition", prefix)
    balanced_keys = {row_key(row) for row in release["balanced20_all"]}
    if not balanced_keys <= {row_key(row) for row in release["mixed_all"]}:
        fail("balanced_not_subset_of_mixed", None)
    if {row_key(row) for row in grounded} - balanced_keys:
        fail("balanced_missing_grounded_rows", None)

    # No design/source lineage may cross train and validation in any release.
    lineage_checks = {}
    split_consistency: dict[str, set[str]] = {}
    for prefix in ("grounded", "mixed", "balanced20"):
        train_groups = {ASSEMBLE.group_key(row) for row in release[f"{prefix}_train"]}
        val_groups = {ASSEMBLE.group_key(row) for row in release[f"{prefix}_validation"]}
        overlap = sorted(train_groups & val_groups)
        lineage_checks[prefix] = {
            "train_groups": len(train_groups), "validation_groups": len(val_groups), "overlap": overlap
        }
        if overlap:
            fail("train_validation_lineage_overlap", {"artifact": prefix, "groups": overlap})
        for row in release[f"{prefix}_all"]:
            expected_group_hash = hashlib.sha256(ASSEMBLE.group_key(row).encode()).hexdigest()
            if row.get("split_group_sha256") != expected_group_hash:
                fail("split_group_hash", row["compiler_assignment_id"])
            split_consistency.setdefault(row["compiler_assignment_id"], set()).add(row["split"])
    inconsistent_splits = sorted(key for key, values in split_consistency.items() if len(values) != 1)
    if inconsistent_splits:
        fail("cross_artifact_split_inconsistency", inconsistent_splits)

    # 29 grounded + 116 revised = 145, with 20% grounded in all/train/validation.
    ratio_checks = {}
    for suffix in ("all", "train", "validation"):
        rows = release[f"balanced20_{suffix}"]
        grounded_count = sum(row["corpus_role"] == "grounded" for row in rows)
        ratio_checks[suffix] = {"rows": len(rows), "grounded": grounded_count, "revised": len(rows) - grounded_count}
        if grounded_count * 5 != len(rows):
            fail("balanced20_ratio", {"split": suffix, **ratio_checks[suffix]})
    if ratio_checks != {
        "all": {"rows": 145, "grounded": 29, "revised": 116},
        "train": {"rows": 130, "grounded": 26, "revised": 104},
        "validation": {"rows": 15, "grounded": 3, "revised": 12},
    }:
        fail("balanced20_exact_counts", ratio_checks)

    leakage = ASSEMBLE.eval_leakage(release["mixed_all"], EVALS)
    if leakage["exact_overlap_rows"] or leakage["twelve_gram_overlap_rows"]:
        fail("frozen_eval_leakage", leakage)

    token_results: dict[str, Any] = {}
    if args.skip_token_recompute:
        warnings.append({"code": "token_recompute_skipped", "detail": None})
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        for name, report_path in TOKEN_REPORTS.items():
            stored = read_json(report_path)
            recomputed = token_report(release[name], RELEASE_JSONL[name], tokenizer)
            exact = stored == recomputed
            token_results[name] = {
                "report_path": str(report_path.relative_to(ROOT)),
                "report_sha256": file_hash(report_path),
                "exact_recompute_match": exact,
                "rows": recomputed["rows"],
                "max_tokens": recomputed["full_conversation_tokens"]["max"],
                "rows_over_2048": recomputed["thresholds"]["2048"]["rows_exceeding"],
                "rows_over_4096": recomputed["thresholds"]["4096"]["rows_exceeding"],
            }
            if not exact:
                fail("token_report_recompute_mismatch", name)
            if recomputed["thresholds"]["2048"]["rows_exceeding"] or recomputed["thresholds"]["4096"]["rows_exceeding"]:
                fail("token_limit_exceeded", name)

    inventory = {}
    for path in required:
        if path.exists():
            inventory[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    report = {
        "schema_version": "uae_adab_final_silver_bundle_audit.v1",
        "decision": "authorize_immutable_bundle" if not errors else "block",
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "worklist": len(universe_ids),
            "accepted": len(accepted),
            "dropped": len(dropped),
            "grounded_accepted": len(grounded),
            "revised_accepted": len(revised),
            "reserved_source_groups": len(reserved_groups),
        },
        "checks": {
            "report_hash_count_errors": len(report_hash_errors),
            "wrong_system_messages": len(wrong_system),
            "forced_excluded_versions_present": len(excluded_versions & observed_grounded_versions),
            "eval_exact_overlap_rows": len(leakage["exact_overlap_rows"]),
            "eval_12gram_overlap_rows": len(leakage["twelve_gram_overlap_rows"]),
            "lineage": lineage_checks,
            "balanced20": ratio_checks,
            "tokens": token_results,
        },
        "bundle_roots": [str(INPUT_DIR.relative_to(ROOT)), str(RELEASE_DIR.relative_to(ROOT))],
        "inventory": inventory,
        "bundle_manifest_sha256": canonical_hash(inventory),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "inventory"}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
