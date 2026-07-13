#!/usr/bin/env python3
"""Validate rights-safe UAE Adab SFT JSONL datasets.

The JSON Schema is the portable field contract. This script additionally enforces
cross-field, dataset-wide, provenance, and held-out exclusion rules without
requiring third-party Python packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


SCHEMA_VERSION = "1.0.0"
ID_RE = re.compile(r"^uae_adab_[a-z0-9][a-z0-9_-]{5,95}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,49}$")

ALLOWED_RIGHTS_CLASSES = {
    "project_owned",
    "commissioned_with_release",
    "licensed_permissive",
    "public_domain",
    "synthetic_rights_safe",
}
DENIED_RIGHTS_CLASSES = {
    "research_only",
    "noncommercial",
    "no_derivatives",
    "unknown",
    "third_party_unlicensed",
    "heldout_eval_only",
}
ALL_RIGHTS_CLASSES = ALLOWED_RIGHTS_CLASSES | DENIED_RIGHTS_CLASSES
SOURCE_TYPES = {
    "original_human_authored",
    "commissioned_human_authored",
    "synthetic",
    "transformed_permissive",
    "public_domain",
}
LANGUAGES = {"en", "en-AE", "ar", "ar-AE"}
REGISTERS = {"institutional_light", "family_rich"}
GRADE_BANDS = {"primary_1_5", "middle_6_8", "secondary_9_12", "adult"}
SPLITS = {"train", "validation"}
QUALITY_CHECKS = {
    "academic_correctness",
    "adab_embodied",
    "non_humiliating_correction",
    "integrity_and_anti_cheating",
    "no_fatwa_or_sectarian_claim",
    "civic_floor",
    "deletion_test",
    "not_answer_only",
    "privacy_safe",
}
TOP_LEVEL_FIELDS = {
    "schema_version",
    "id",
    "split",
    "language",
    "register",
    "subject",
    "grade_band",
    "learning_objective",
    "scenario_family",
    "pressure_pattern",
    "tags",
    "messages",
    "conversation_sha256",
    "provenance",
    "review",
}
REQUIRED_TOP_LEVEL_FIELDS = TOP_LEVEL_FIELDS - {"tags"}
PROVENANCE_FIELDS = {
    "source_type",
    "source_ids",
    "source_urls",
    "derivation",
    "rights_class",
    "rights_holder",
    "license_id",
    "license_url",
    "permission_scope",
    "rights_verified_by",
    "rights_verified_at",
    "contains_verbatim_third_party_text",
    "personal_data_status",
    "consent_record_id",
    "input_rights_classes",
    "generator",
}
REQUIRED_PROVENANCE_FIELDS = PROVENANCE_FIELDS - {"consent_record_id", "generator"}
GENERATOR_FIELDS = {"provider", "model", "job_id", "generated_at", "terms_url"}
REVIEW_FIELDS = {"status", "reviewers", "reviewed_at", "notes", "quality_checks"}
REQUIRED_REVIEW_FIELDS = REVIEW_FIELDS - {"notes"}


def normalize_text(value: str) -> str:
    """Normalize Unicode and insignificant line whitespace for stable exact hashes."""
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in value.split("\n")).strip()


def text_sha256(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def canonical_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": message["role"], "content": normalize_text(message["content"])}
        for message in messages
    ]


def compute_conversation_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        canonical_messages(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_nonempty_string(value: Any, minimum: int = 1, maximum: int | None = None) -> bool:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        return False
    return maximum is None or len(value) <= maximum


def is_uri(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_datetime(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = value.replace("Z", "+00:00") if value.endswith("Z") else value
        datetime.fromisoformat(parsed)
        return "T" in value
    except ValueError:
        return False


def _extract_path_values(value: Any, path: str) -> list[Any]:
    """Extract values for a small manifest path syntax such as turns[].user."""
    values = [value]
    for part in path.split("."):
        expand = part.endswith("[]")
        key = part[:-2] if expand else part
        next_values: list[Any] = []
        for current in values:
            if not isinstance(current, dict) or key not in current:
                continue
            found = current[key]
            if expand:
                if isinstance(found, list):
                    next_values.extend(found)
            else:
                next_values.append(found)
        values = next_values
    return values


def load_heldout_exclusions(
    manifest_path: Path, repo_root: Path
) -> tuple[set[str], set[str], list[dict[str, Any]], dict[str, int]]:
    """Load and verify pinned held-out sources, returning IDs and normalized text hashes."""
    issues: list[dict[str, Any]] = []
    ids: set[str] = set()
    hashes: set[str] = set()

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load held-out manifest {manifest_path}: {exc}") from exc

    for value in manifest.get("ids", []):
        ids.add(str(value))
    for value in manifest.get("content_hashes", []):
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            issues.append(
                {"code": "heldout_manifest_invalid_hash", "message": f"invalid hash: {value!r}"}
            )
        else:
            hashes.add(value)

    source_count = 0
    for source in manifest.get("sources", []):
        source_count += 1
        relative_path = source.get("repo_relative_path")
        namespace = source.get("namespace")
        if not is_nonempty_string(relative_path) or not is_nonempty_string(namespace):
            issues.append(
                {
                    "code": "heldout_manifest_invalid_source",
                    "message": "source requires namespace and repo_relative_path",
                }
            )
            continue
        path = repo_root / relative_path
        if not path.is_file():
            issues.append(
                {
                    "code": "heldout_source_missing",
                    "message": f"held-out source is missing: {relative_path}",
                }
            )
            continue
        actual_file_hash = file_sha256(path)
        if actual_file_hash != source.get("file_sha256"):
            issues.append(
                {
                    "code": "heldout_source_hash_mismatch",
                    "message": (
                        f"held-out source changed: {relative_path}; expected "
                        f"{source.get('file_sha256')}, got {actual_file_hash}"
                    ),
                }
            )
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                {
                    "code": "heldout_source_invalid_json",
                    "message": f"cannot parse {relative_path}: {exc}",
                }
            )
            continue
        if not isinstance(rows, list):
            rows = [rows]
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_id = row.get(source.get("id_field", "id"))
            if raw_id is not None:
                ids.add(f"{namespace}:{raw_id}")
                if source.get("include_raw_string_ids") and isinstance(raw_id, str):
                    ids.add(raw_id)
            for text_path in source.get("text_paths", []):
                for text in _extract_path_values(row, text_path):
                    if isinstance(text, str) and normalize_text(text):
                        hashes.add(text_sha256(text))

    stats = {
        "heldout_sources_loaded": source_count,
        "heldout_ids_loaded": len(ids),
        "heldout_content_hashes_loaded": len(hashes),
    }
    return ids, hashes, issues, stats


def _add_issue(
    issues: list[dict[str, Any]],
    code: str,
    message: str,
    line: int,
    record_id: str | None,
) -> None:
    issues.append({"code": code, "message": message, "line": line, "record_id": record_id})


def _check_fields(
    obj: dict[str, Any],
    allowed: set[str],
    required: set[str],
    path: str,
    add: Any,
) -> None:
    missing = sorted(required - obj.keys())
    extra = sorted(obj.keys() - allowed)
    if missing:
        add("missing_fields", f"{path} missing required fields: {', '.join(missing)}")
    if extra:
        add("unknown_fields", f"{path} has unknown fields: {', '.join(extra)}")


def validate_record(
    record: Any,
    line: int,
    heldout_ids: set[str],
    heldout_hashes: set[str],
) -> tuple[list[dict[str, Any]], str | None]:
    issues: list[dict[str, Any]] = []
    record_id = record.get("id") if isinstance(record, dict) and isinstance(record.get("id"), str) else None

    def add(code: str, message: str) -> None:
        _add_issue(issues, code, message, line, record_id)

    if not isinstance(record, dict):
        add("record_not_object", "row must be a JSON object")
        return issues, None

    _check_fields(record, TOP_LEVEL_FIELDS, REQUIRED_TOP_LEVEL_FIELDS, "record", add)

    if record.get("schema_version") != SCHEMA_VERSION:
        add("schema_version", f"schema_version must be {SCHEMA_VERSION!r}")
    if not isinstance(record_id, str) or not ID_RE.fullmatch(record_id):
        add("invalid_id", "id must match ^uae_adab_[a-z0-9][a-z0-9_-]{5,95}$")
    elif record_id in heldout_ids:
        add("heldout_id_match", f"record id is reserved by held-out data: {record_id}")
    if record.get("split") not in SPLITS:
        add("invalid_split", f"split must be one of {sorted(SPLITS)}")
    if record.get("language") not in LANGUAGES:
        add("invalid_language", f"language must be one of {sorted(LANGUAGES)}")
    if record.get("register") not in REGISTERS:
        add("invalid_register", f"register must be one of {sorted(REGISTERS)}")
    if record.get("grade_band") not in GRADE_BANDS:
        add("invalid_grade_band", f"grade_band must be one of {sorted(GRADE_BANDS)}")
    for field, minimum, maximum in (
        ("subject", 2, 80),
        ("learning_objective", 10, 500),
        ("scenario_family", 3, 100),
    ):
        if not is_nonempty_string(record.get(field), minimum, maximum):
            add("invalid_text_field", f"{field} must be a non-empty string of length {minimum}-{maximum}")

    tags = record.get("tags", [])
    if not isinstance(tags, list) or len(tags) > 30 or any(
        not isinstance(tag, str) or not TAG_RE.fullmatch(tag) for tag in tags
    ):
        add("invalid_tags", "tags must be at most 30 lowercase slug strings")
    elif len(tags) != len(set(tags)):
        add("duplicate_tags", "tags must be unique")

    messages = record.get("messages")
    user_count = 0
    calculated_hash: str | None = None
    if not isinstance(messages, list) or not 3 <= len(messages) <= 41:
        add("invalid_messages", "messages must be a list with 3-41 items")
    else:
        expected_role = "system"
        messages_well_formed = True
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                add("invalid_message", f"messages[{index}] must be an object")
                messages_well_formed = False
                continue
            if set(message) != {"role", "content"}:
                add("invalid_message_fields", f"messages[{index}] must contain only role and content")
                messages_well_formed = False
            role = message.get("role")
            content = message.get("content")
            if role != expected_role:
                add(
                    "invalid_role_sequence",
                    f"messages[{index}].role must be {expected_role!r}, got {role!r}",
                )
            if role == "user":
                user_count += 1
            if not is_nonempty_string(content, 1, 20000):
                add("invalid_message_content", f"messages[{index}].content must be non-empty")
                messages_well_formed = False
            elif text_sha256(content) in heldout_hashes:
                add(
                    "heldout_content_match",
                    f"messages[{index}].content exactly matches normalized held-out text",
                )
            if index == 0:
                expected_role = "user"
            elif expected_role == "user":
                expected_role = "assistant"
            else:
                expected_role = "user"
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") != "assistant":
            add("conversation_not_closed", "conversation must end with an assistant message")
        if messages_well_formed:
            calculated_hash = compute_conversation_hash(messages)
            supplied_hash = record.get("conversation_sha256")
            if not isinstance(supplied_hash, str) or not SHA256_RE.fullmatch(supplied_hash):
                add("invalid_conversation_hash", "conversation_sha256 must be 64 lowercase hex characters")
            elif supplied_hash != calculated_hash:
                add(
                    "conversation_hash_mismatch",
                    f"conversation_sha256 must be {calculated_hash} for these messages",
                )
            if calculated_hash in heldout_hashes:
                add("heldout_conversation_match", "full conversation hash matches held-out content")

    pressure = record.get("pressure_pattern")
    if not isinstance(pressure, list) or not pressure or len(pressure) > 20 or any(
        type(score) is not int or not 0 <= score <= 5 for score in pressure
    ):
        add("invalid_pressure_pattern", "pressure_pattern must contain 1-20 integers from 0 through 5")
    elif isinstance(messages, list) and len(pressure) != user_count:
        add(
            "pressure_turn_mismatch",
            f"pressure_pattern has {len(pressure)} scores but conversation has {user_count} user messages",
        )

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        add("invalid_provenance", "provenance must be an object")
    else:
        _check_fields(
            provenance,
            PROVENANCE_FIELDS,
            REQUIRED_PROVENANCE_FIELDS,
            "provenance",
            add,
        )
        source_type = provenance.get("source_type")
        rights_class = provenance.get("rights_class")
        input_rights = provenance.get("input_rights_classes")
        source_ids = provenance.get("source_ids")
        if source_type not in SOURCE_TYPES:
            add("invalid_source_type", f"source_type must be one of {sorted(SOURCE_TYPES)}")
        if rights_class not in ALL_RIGHTS_CLASSES:
            add("invalid_rights_class", f"unknown rights_class: {rights_class!r}")
        elif rights_class in DENIED_RIGHTS_CLASSES:
            add("rights_not_trainable", f"rights_class {rights_class!r} is not trainable")
        if not isinstance(input_rights, list) or not input_rights:
            add("invalid_input_rights", "input_rights_classes must be a non-empty list")
        else:
            unknown = sorted({value for value in input_rights if value not in ALL_RIGHTS_CLASSES})
            denied = sorted(set(input_rights) & DENIED_RIGHTS_CLASSES)
            if unknown:
                add("invalid_input_rights", f"unknown input rights classes: {', '.join(unknown)}")
            if denied:
                add("input_rights_not_trainable", f"prohibited input rights classes: {', '.join(denied)}")
        if not isinstance(source_ids, list) or not source_ids or any(
            not is_nonempty_string(value, 2, 200) for value in source_ids
        ):
            add("invalid_source_ids", "source_ids must be a non-empty list of stable strings")
        else:
            if len(source_ids) != len(set(source_ids)):
                add("duplicate_source_ids", "source_ids must be unique")
            matches = sorted(set(source_ids) & heldout_ids)
            if matches:
                add("heldout_source_id_match", f"source_ids contain held-out IDs: {', '.join(matches)}")
        source_urls = provenance.get("source_urls")
        if not isinstance(source_urls, list) or any(not is_uri(url) for url in source_urls):
            add("invalid_source_urls", "source_urls must be a list of HTTP(S) URLs")
        elif len(source_urls) != len(set(source_urls)):
            add("duplicate_source_urls", "source_urls must be unique")
        for field, minimum, maximum in (
            ("derivation", 10, 2000),
            ("rights_holder", 2, 300),
            ("license_id", 2, 100),
            ("rights_verified_by", 2, 200),
        ):
            if not is_nonempty_string(provenance.get(field), minimum, maximum):
                add("invalid_provenance_field", f"provenance.{field} is missing or invalid")
        if not is_uri(provenance.get("license_url")):
            add("invalid_license_url", "provenance.license_url must be an HTTP(S) URL")
        if provenance.get("permission_scope") != "commercial_training_and_dataset_distribution":
            add(
                "invalid_permission_scope",
                "permission_scope must allow commercial training and dataset distribution",
            )
        if not is_datetime(provenance.get("rights_verified_at")):
            add("invalid_rights_verified_at", "rights_verified_at must be an ISO 8601 date-time")
        if provenance.get("contains_verbatim_third_party_text") is not False:
            add("verbatim_third_party_text", "contains_verbatim_third_party_text must be false")
        personal_status = provenance.get("personal_data_status")
        if personal_status not in {"none", "anonymized_with_consent"}:
            add("invalid_personal_data_status", "personal_data_status is invalid")
        needs_consent = rights_class == "commissioned_with_release" or personal_status == "anonymized_with_consent"
        if needs_consent and not is_nonempty_string(provenance.get("consent_record_id"), 2, 200):
            add("missing_consent_record", "commissioned or consented personal data requires consent_record_id")
        if rights_class == "licensed_permissive" and (
            provenance.get("license_id") in {"unknown", "custom-unknown"}
            or not is_uri(provenance.get("license_url"))
        ):
            add("unverified_permissive_license", "licensed_permissive requires an explicit license")
        generator = provenance.get("generator")
        if source_type == "synthetic" or rights_class == "synthetic_rights_safe":
            if not isinstance(generator, dict):
                add("missing_generator", "synthetic records require provenance.generator")
            else:
                _check_fields(generator, GENERATOR_FIELDS, GENERATOR_FIELDS, "provenance.generator", add)
                for field in ("provider", "model", "job_id"):
                    if not is_nonempty_string(generator.get(field), 2, 200):
                        add("invalid_generator", f"generator.{field} is missing or invalid")
                if not is_datetime(generator.get("generated_at")):
                    add("invalid_generator", "generator.generated_at must be an ISO 8601 date-time")
                if not is_uri(generator.get("terms_url")):
                    add("invalid_generator", "generator.terms_url must be an HTTP(S) URL")
        elif generator is not None:
            add("unexpected_generator", "generator is only valid for synthetic records")

    review = record.get("review")
    if not isinstance(review, dict):
        add("invalid_review", "review must be an object")
    else:
        _check_fields(review, REVIEW_FIELDS, REQUIRED_REVIEW_FIELDS, "review", add)
        if review.get("status") != "approved":
            add("review_not_approved", "review.status must be approved")
        reviewers = review.get("reviewers")
        if not isinstance(reviewers, list) or not reviewers or any(
            not is_nonempty_string(value, 2, 200) for value in reviewers
        ):
            add("invalid_reviewers", "reviewers must be a non-empty list")
        elif len(reviewers) != len(set(reviewers)):
            add("duplicate_reviewers", "reviewers must be unique")
        if not is_datetime(review.get("reviewed_at")):
            add("invalid_reviewed_at", "reviewed_at must be an ISO 8601 date-time")
        quality = review.get("quality_checks")
        if not isinstance(quality, dict):
            add("invalid_quality_checks", "quality_checks must be an object")
        else:
            if set(quality) != QUALITY_CHECKS:
                missing = sorted(QUALITY_CHECKS - quality.keys())
                extra = sorted(quality.keys() - QUALITY_CHECKS)
                add(
                    "invalid_quality_check_fields",
                    f"quality checks missing={missing} extra={extra}",
                )
            failed = sorted(key for key in QUALITY_CHECKS if quality.get(key) is not True)
            if failed:
                add("quality_checks_failed", f"quality checks not true: {', '.join(failed)}")

    return issues, calculated_hash


def validate_dataset(
    dataset_path: Path,
    exclusions_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    heldout_ids, heldout_hashes, manifest_issues, heldout_stats = load_heldout_exclusions(
        exclusions_path, repo_root
    )
    errors: list[dict[str, Any]] = list(manifest_issues)
    warnings: list[dict[str, Any]] = []
    rows: list[tuple[int, Any, str | None, list[dict[str, Any]]]] = []
    counters: dict[str, Counter[str]] = {
        key: Counter()
        for key in ("split", "language", "register", "rights_class", "subject", "scenario_family")
    }
    input_lines = 0
    parsed_rows = 0

    try:
        handle = dataset_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot open dataset {dataset_path}: {exc}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            input_lines += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                issue = {
                    "code": "invalid_json",
                    "message": str(exc),
                    "line": line_number,
                    "record_id": None,
                }
                errors.append(issue)
                rows.append((line_number, None, None, [issue]))
                continue
            parsed_rows += 1
            row_issues, calculated_hash = validate_record(
                record, line_number, heldout_ids, heldout_hashes
            )
            errors.extend(row_issues)
            rows.append((line_number, record, calculated_hash, row_issues))
            if isinstance(record, dict):
                for field in ("split", "language", "register", "subject", "scenario_family"):
                    value = record.get(field)
                    if isinstance(value, str):
                        counters[field][value] += 1
                provenance = record.get("provenance")
                if isinstance(provenance, dict) and isinstance(provenance.get("rights_class"), str):
                    counters["rights_class"][provenance["rights_class"]] += 1

    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}
    duplicate_groups: list[dict[str, Any]] = []
    row_issue_lines: set[int] = {issue.get("line") for issue in errors if issue.get("line") is not None}
    for line_number, record, calculated_hash, _ in rows:
        if not isinstance(record, dict):
            continue
        record_id = record.get("id")
        if isinstance(record_id, str):
            if record_id in seen_ids:
                issue = {
                    "code": "duplicate_id",
                    "message": f"id duplicates line {seen_ids[record_id]}: {record_id}",
                    "line": line_number,
                    "record_id": record_id,
                }
                errors.append(issue)
                row_issue_lines.add(line_number)
                duplicate_groups.append(
                    {"kind": "id", "value": record_id, "lines": [seen_ids[record_id], line_number]}
                )
            else:
                seen_ids[record_id] = line_number
        if calculated_hash:
            if calculated_hash in seen_hashes:
                issue = {
                    "code": "duplicate_conversation",
                    "message": f"exact normalized messages duplicate line {seen_hashes[calculated_hash]}",
                    "line": line_number,
                    "record_id": record_id,
                }
                errors.append(issue)
                row_issue_lines.add(line_number)
                duplicate_groups.append(
                    {
                        "kind": "conversation_sha256",
                        "value": calculated_hash,
                        "lines": [seen_hashes[calculated_hash], line_number],
                    }
                )
            else:
                seen_hashes[calculated_hash] = line_number

    invalid_rows = len({line for line in row_issue_lines if isinstance(line, int)})
    valid_rows = max(0, input_lines - invalid_rows)
    heldout_matches = sum(1 for issue in errors if issue["code"].startswith("heldout_"))

    return {
        "dataset": str(dataset_path),
        "schema_version": SCHEMA_VERSION,
        "ok": not errors,
        "input_lines": input_lines,
        "parsed_rows": parsed_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "exact_duplicate_groups": duplicate_groups,
        "heldout_matches": heldout_matches,
        **heldout_stats,
        "counts": {key: dict(sorted(counter.items())) for key, counter in counters.items()},
        "errors": errors,
        "warnings": warnings,
    }


def _load_hashable_records(path: Path) -> Iterable[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value if isinstance(value, list) else [value]
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
            raise ValueError("every JSON object must contain a messages list")
        yield record


def print_human_report(report: dict[str, Any]) -> None:
    status = "PASS" if report["ok"] else "FAIL"
    print(
        f"{status}: {report['dataset']} — {report['valid_rows']} valid, "
        f"{report['invalid_rows']} invalid, {report['error_count']} errors"
    )
    print(
        f"Held-out exclusions: {report['heldout_ids_loaded']} IDs, "
        f"{report['heldout_content_hashes_loaded']} content hashes, "
        f"{report['heldout_matches']} matches"
    )
    if report["exact_duplicate_groups"]:
        print(f"Exact duplicate groups: {len(report['exact_duplicate_groups'])}")
    for key, values in report["counts"].items():
        if values:
            rendered = ", ".join(f"{name}={count}" for name, count in values.items())
            print(f"{key}: {rendered}")
    for issue in report["errors"]:
        location = f"line {issue['line']}" if issue.get("line") else "dataset"
        print(f"ERROR [{issue['code']}] {location}: {issue['message']}")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, help="JSONL dataset to validate")
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=repo_root / "data/uae_adab/heldout_exclusions.json",
        help="held-out exclusion manifest",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--json", action="store_true", help="print machine-readable report")
    parser.add_argument(
        "--print-hash",
        type=Path,
        metavar="JSON",
        help="print canonical conversation SHA-256 for a JSON object or array",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_hash:
        try:
            for record in _load_hashable_records(args.print_hash):
                print(compute_conversation_hash(record["messages"]))
            return 0
        except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    if args.dataset is None:
        print("ERROR: dataset is required unless --print-hash is used", file=sys.stderr)
        return 2
    try:
        report = validate_dataset(args.dataset, args.exclusions, args.repo_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_human_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
