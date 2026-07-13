#!/usr/bin/env python3
"""Deterministic content-risk gate for UAE Adab conversation JSONL.

The gate is deliberately non-mutating: it emits reviewable flags and never
repairs, filters, or rewrites source records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


GATE_VERSION = "1.0.0"
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class EvalReference:
    source: str
    pointer: str
    text: str


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("\u2019", "'")
    return " ".join(WORD_RE.findall(value))


def tokens(value: str) -> list[str]:
    return WORD_RE.findall(unicodedata.normalize("NFKC", value).casefold().replace("\u2019", "'"))


def excerpt(value: str, limit: int = 220) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _walk_strings(value: Any, pointer: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield pointer or "/", value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{pointer}/{index}")
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _walk_strings(value[key], f"{pointer}/{key}")


def load_eval_references(paths: Sequence[Path]) -> list[EvalReference]:
    references: list[EvalReference] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for pointer, value in _walk_strings(raw):
            if len(tokens(value)) >= 5:
                references.append(EvalReference(str(path), pointer, value))
    return references


def default_eval_paths(repo_root: Path) -> list[Path]:
    manifest = repo_root / "data/uae_adab/heldout_exclusions.json"
    if not manifest.is_file():
        return []
    value = json.loads(manifest.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for source in value.get("sources", []):
        relative = source.get("repo_relative_path")
        if isinstance(relative, str) and (repo_root / relative).is_file():
            paths.append(repo_root / relative)
    return paths


def _messages(record: Any) -> list[dict[str, str]]:
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        return []
    return [
        message
        for message in record["messages"]
        if isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
    ]


def _record_id(record: Any, line: int) -> str:
    if isinstance(record, dict):
        for key in ("id", "skeleton_id", "case_id"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"line_{line}"


def _flag(
    code: str,
    severity: str,
    category: str,
    evidence: str,
    message_indices: Iterable[int] = (),
    **details: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "category": category,
        "scope": "record",
        "message_indices": sorted(set(message_indices)),
        "evidence": excerpt(evidence),
    }
    if details:
        result["details"] = details
    return result


# All patterns apply only to assistant messages. Their names describe signals,
# not conclusions; ambiguous religious language is intentionally review-only.
ASSISTANT_PATTERNS: tuple[tuple[str, str, str, re.Pattern[str]], ...] = (
    (
        "scripture_or_hadith_quote_signal",
        "block",
        "religious_boundary",
        re.compile(
            r"\b(?:qur['’]?an|surah|sura|ayah|verse)\s*(?:says|states|teaches|\d+\s*[:.]\s*\d+)"
            r"|\b(?:the\s+)?prophet\s+(?:muhammad\s+)?(?:said|taught)\b"
            r"|\b(?:narrated\s+by|sahih\s+(?:al-)?bukhari|sahih\s+muslim|hadith\s+(?:says|states))\b"
            r"|\bpeace\s+be\s+upon\s+him\b|\(صلى\s+الله\s+عليه\s+وسلم\)",
            re.I,
        ),
    ),
    (
        "fatwa_or_religious_ruling_signal",
        "block",
        "religious_boundary",
        re.compile(
            r"\b(?:this|that|it|doing\s+this|you)\s+(?:is|are)\s+(?:clearly\s+|definitely\s+)?"
            r"(?:halal|haram|sinful|a\s+sin|forbidden\s+in\s+islam|obligatory\s+in\s+islam)\b"
            r"|\bi\s+(?:rule|declare)\s+(?:that\s+)?(?:it|this|that)\s+(?:halal|haram)\b",
            re.I,
        ),
    ),
    (
        "fatwa_or_sectarian_term_signal",
        "review",
        "religious_boundary",
        re.compile(r"\b(?:fatwa|sunni|shia|shi['’]?a|salafi|sufi|wahhabi|takfir|kafir)\b", re.I),
    ),
    (
        "sectarian_language_signal",
        "review",
        "religious_boundary",
        re.compile(
            r"\b(?:sunni|shia|shi['’]?a|salafi|sufi|wahhabi)\b.*\b(?:wrong|deviant|superior|inferior|true\s+muslim)\b"
            r"|\b(?:wrong|deviant|superior|inferior)\b.*\b(?:sunni|shia|shi['’]?a|salafi|sufi|wahhabi)\b",
            re.I,
        ),
    ),
    (
        "humiliation_or_contempt_signal",
        "block",
        "dignity",
        re.compile(
            r"\b(?:you|your\s+(?:teacher|classmate|father|mother|friend))\s+(?:are|is)\s+"
            r"(?:an?\s+)?(?:stupid|idiot|hopeless|pathetic|useless|moron|failure)\b"
            r"|\bwhat\s+(?:an?\s+)?(?:idiotic|stupid|pathetic)\s+(?:answer|mistake|question)\b",
            re.I,
        ),
    ),
    (
        "stereotype_or_caricature_signal",
        "block",
        "civic_floor",
        re.compile(
            r"\b(?:all|most)\s+(?:emiratis|muslims|arabs|westerners|expats)\s+(?:are|always|never|cannot|can't)\b"
            r"|\b(?:emiratis|muslims|arabs|westerners|expats)\s+are\s+(?:naturally|inherently|typically)\b"
            r"|\b(?:emiratis|muslims|arabs|westerners|expats)\s+are\s+"
            r"(?:dishonest|lazy|violent|backward|uncivilized|inferior|superior|terrorists)\b",
            re.I,
        ),
    ),
    (
        "cheating_or_answer_leakage_signal",
        "block",
        "academic_integrity",
        re.compile(
            r"\b(?:copy\s+and\s+paste|submit|hand\s+in|turn\s+in)\s+(?:this|the\s+following)\s+(?:as\s+)?(?:your\s+)?(?:own\s+)?answer\b"
            r"|\bhere(?:'s|\s+is)\s+(?:the\s+)?(?:completed|ready-to-submit|full)\s+(?:assignment|essay|test|worksheet)\b"
            r"|\bwrite\s+exactly\s+this\s+(?:on|in)\s+your\s+(?:test|exam|quiz|assignment)\b"
            r"|\bhere(?:'s|\s+is)\s+(?:your|the)\s+final\s+answer\s+to\s+(?:submit|hand\s+in|turn\s+in)\b"
            r"|\banswer\s+key\s*:\s*",
            re.I,
        ),
    ),
    (
        "assistant_role_leakage_signal",
        "review",
        "role_integrity",
        re.compile(
            r"<adab_register>|\bsymbolic\s+lesson\s+skeleton\b|\brealization_notes\b"
            r"|\bmy\s+system\s+prompt\s+(?:says|instructs|tells)\b"
            r"|\baccording\s+to\s+(?:the|my)\s+system\s+prompt\b"
            r"|\bi\s+was\s+instructed\s+to\s+(?:generate|write|act|respond)\b"
            r"|\bas\s+(?:an\s+)?ai\s+(?:language\s+model|assistant)\b",
            re.I,
        ),
    ),
)

DECORATION_RE = re.compile(
    r"\b(?:alhamdulillah|mashallah|inshallah|subhanallah|jazakallah(?:u\s+khairan)?|"
    r"barakallahu\s+feek|bismillah|ameen|allah\s+willing)\b",
    re.I,
)


def _decision(flags: Sequence[dict[str, Any]]) -> str:
    if any(flag["severity"] == "block" for flag in flags):
        return "block"
    if flags:
        return "review"
    return "pass"


def _eval_index(
    references: Sequence[EvalReference], n: int
) -> tuple[dict[tuple[str, ...], list[tuple[int, int]]], list[list[str]]]:
    index: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    reference_tokens: list[list[str]] = []
    for ref_index, reference in enumerate(references):
        words = tokens(reference.text)
        reference_tokens.append(words)
        for start in range(len(words) - n + 1):
            index[tuple(words[start : start + n])].append((ref_index, start))
    return index, reference_tokens


def _longest_eval_overlap(
    content: str,
    references: Sequence[EvalReference],
    index: dict[tuple[str, ...], list[tuple[int, int]]],
    reference_tokens: Sequence[list[str]],
    n: int,
) -> tuple[int, EvalReference | None, str]:
    words = tokens(content)
    best = (0, None, "")
    for start in range(len(words) - n + 1):
        key = tuple(words[start : start + n])
        for ref_index, ref_start in index.get(key, []):
            length = n
            while (
                start + length < len(words)
                and ref_start + length < len(reference_tokens[ref_index])
                and words[start + length] == reference_tokens[ref_index][ref_start + length]
            ):
                length += 1
            if length > best[0]:
                best = (
                    length,
                    references[ref_index],
                    " ".join(words[start : start + length]),
                )
    return best


def _shingles(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def analyze_records(
    records: Sequence[Any],
    eval_references: Sequence[EvalReference] = (),
    *,
    eval_overlap_words: int = 8,
    verbatim_span_words: int = 16,
    opening_words: int = 8,
    opening_repeat_min: int = 3,
    near_duplicate_threshold: float = 0.82,
) -> dict[str, Any]:
    """Return deterministic record- and corpus-level risk flags."""
    results: list[dict[str, Any]] = []
    eval_index, reference_tokens = _eval_index(eval_references, eval_overlap_words)

    for line, record in enumerate(records, start=1):
        record_id = _record_id(record, line)
        flags: list[dict[str, Any]] = []
        messages = _messages(record)
        if not messages:
            flags.append(
                _flag(
                    "unscannable_messages",
                    "block",
                    "input_integrity",
                    "Record has no well-formed messages to scan.",
                )
            )
        for message_index, message in enumerate(messages):
            content = message["content"]
            if eval_references and len(tokens(content)) >= eval_overlap_words:
                length, reference, span = _longest_eval_overlap(
                    content,
                    eval_references,
                    eval_index,
                    reference_tokens,
                    eval_overlap_words,
                )
                if reference is not None and length >= eval_overlap_words:
                    flags.append(
                        _flag(
                            "heldout_eval_phrase_overlap",
                            "block",
                            "evaluation_contamination",
                            span,
                            [message_index],
                            overlap_words=length,
                            eval_source=reference.source,
                            eval_pointer=reference.pointer,
                        )
                    )
            if message["role"] != "assistant":
                continue
            for code, severity, category, pattern in ASSISTANT_PATTERNS:
                match = pattern.search(content)
                if match:
                    flags.append(
                        _flag(code, severity, category, match.group(0), [message_index])
                    )
            decorations = list(DECORATION_RE.finditer(content))
            if len(decorations) >= 3:
                flags.append(
                    _flag(
                        "excessive_religious_decoration_signal",
                        "review",
                        "naturalness",
                        ", ".join(match.group(0) for match in decorations),
                        [message_index],
                        decoration_count=len(decorations),
                    )
                )
        flags.sort(key=lambda item: (item["code"], item["message_indices"], item["evidence"]))
        results.append(
            {
                "line": line,
                "record_id": record_id,
                "decision": _decision(flags),
                "flags": flags,
            }
        )

    corpus_flags: list[dict[str, Any]] = []
    by_result = {result["record_id"]: result for result in results}

    # Repeated assistant openings.
    openings: dict[tuple[str, ...], list[tuple[str, int]]] = defaultdict(list)
    for line, record in enumerate(records, start=1):
        record_id = _record_id(record, line)
        for message_index, message in enumerate(_messages(record)):
            if message["role"] == "assistant":
                words = tokens(message["content"])
                if len(words) >= opening_words:
                    openings[tuple(words[:opening_words])].append((record_id, message_index))
    for opening, occurrences in sorted(openings.items()):
        distinct = sorted({record_id for record_id, _ in occurrences})
        if len(distinct) < opening_repeat_min:
            continue
        corpus_flag = {
            "code": "repeated_templated_opening",
            "severity": "review",
            "category": "template_repetition",
            "scope": "corpus",
            "record_ids": distinct,
            "evidence": " ".join(opening),
            "details": {"occurrence_count": len(occurrences), "opening_words": opening_words},
        }
        corpus_flags.append(corpus_flag)
        for record_id, message_index in occurrences:
            by_result[record_id]["flags"].append(
                _flag(
                    "repeated_templated_opening",
                    "review",
                    "template_repetition",
                    " ".join(opening),
                    [message_index],
                    corpus_occurrence_count=len(occurrences),
                )
            )

    # Suspicious shared long spans across records.
    span_index: dict[tuple[str, ...], list[tuple[str, int]]] = defaultdict(list)
    for line, record in enumerate(records, start=1):
        record_id = _record_id(record, line)
        seen_in_record: set[tuple[str, ...]] = set()
        for message_index, message in enumerate(_messages(record)):
            words = tokens(message["content"])
            for span in _shingles(words, verbatim_span_words):
                if span not in seen_in_record:
                    span_index[span].append((record_id, message_index))
                    seen_in_record.add(span)
    span_groups: dict[tuple[str, ...], list[tuple[str, int]]] = {}
    for span, occurrences in span_index.items():
        if len({record_id for record_id, _ in occurrences}) >= 2:
            span_groups[span] = occurrences
    # Report maximal-ish spans only: suppress a span whose first n-1 suffix is
    # already represented with the same occurrence set.
    reported_occurrence_sets: set[tuple[tuple[str, int], ...]] = set()
    for span, occurrences in sorted(span_groups.items()):
        occurrence_key = tuple(sorted(set(occurrences)))
        if occurrence_key in reported_occurrence_sets:
            continue
        reported_occurrence_sets.add(occurrence_key)
        distinct = sorted({record_id for record_id, _ in occurrences})
        corpus_flags.append(
            {
                "code": "suspicious_shared_verbatim_span",
                "severity": "review",
                "category": "verbatim_overlap",
                "scope": "corpus",
                "record_ids": distinct,
                "evidence": " ".join(span),
                "details": {"span_words": verbatim_span_words},
            }
        )
        for record_id, message_index in sorted(set(occurrences)):
            by_result[record_id]["flags"].append(
                _flag(
                    "suspicious_shared_verbatim_span",
                    "review",
                    "verbatim_overlap",
                    " ".join(span),
                    [message_index],
                    peer_record_ids=[item for item in distinct if item != record_id],
                    span_words=verbatim_span_words,
                )
            )

    # Exact and near corpus duplicates. System register controls are excluded so
    # they do not make otherwise distinct dialogues look artificially similar.
    conversation_words: dict[str, list[str]] = {}
    conversation_text: dict[str, str] = {}
    for line, record in enumerate(records, start=1):
        record_id = _record_id(record, line)
        parts = [
            f"{message['role']} {normalize_text(message['content'])}"
            for message in _messages(record)
            if message["role"] != "system"
        ]
        text = " ".join(parts)
        conversation_text[record_id] = text
        conversation_words[record_id] = tokens(text)

    exact: dict[str, list[str]] = defaultdict(list)
    for record_id, text in conversation_text.items():
        if text:
            exact[hashlib.sha256(text.encode("utf-8")).hexdigest()].append(record_id)
    exact_pairs: set[tuple[str, str]] = set()
    for digest, ids in sorted(exact.items()):
        distinct = sorted(set(ids))
        if len(distinct) < 2:
            continue
        for left_index, left in enumerate(distinct):
            for right in distinct[left_index + 1 :]:
                exact_pairs.add((left, right))
        corpus_flags.append(
            {
                "code": "exact_corpus_duplicate",
                "severity": "block",
                "category": "corpus_duplication",
                "scope": "corpus",
                "record_ids": distinct,
                "evidence": digest,
                "details": {"duplicate_count": len(distinct)},
            }
        )
        for record_id in distinct:
            by_result[record_id]["flags"].append(
                _flag(
                    "exact_corpus_duplicate",
                    "block",
                    "corpus_duplication",
                    digest,
                    peer_record_ids=[item for item in distinct if item != record_id],
                )
            )

    shingles_by_id = {
        record_id: _shingles(words, 5)
        for record_id, words in conversation_words.items()
        if len(words) >= 10
    }
    postings: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for record_id, values in shingles_by_id.items():
        for value in values:
            postings[value].append(record_id)
    shared_counts: Counter[tuple[str, str]] = Counter()
    for ids in postings.values():
        distinct = sorted(set(ids))
        # Extremely common shingles are boilerplate, not useful pair candidates.
        if len(distinct) > 100:
            continue
        for left_index, left in enumerate(distinct):
            for right in distinct[left_index + 1 :]:
                shared_counts[(left, right)] += 1
    for (left, right), intersection in sorted(shared_counts.items()):
        if (left, right) in exact_pairs:
            continue
        union = len(shingles_by_id[left]) + len(shingles_by_id[right]) - intersection
        similarity = intersection / union if union else 0.0
        if similarity < near_duplicate_threshold:
            continue
        corpus_flags.append(
            {
                "code": "near_corpus_duplicate",
                "severity": "block",
                "category": "corpus_duplication",
                "scope": "corpus",
                "record_ids": [left, right],
                "evidence": f"5-gram Jaccard={similarity:.3f}",
                "details": {"similarity": round(similarity, 6), "threshold": near_duplicate_threshold},
            }
        )
        for record_id, peer in ((left, right), (right, left)):
            by_result[record_id]["flags"].append(
                _flag(
                    "near_corpus_duplicate",
                    "block",
                    "corpus_duplication",
                    f"5-gram Jaccard={similarity:.3f}",
                    peer_record_ids=[peer],
                    similarity=round(similarity, 6),
                )
            )

    code_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for result in results:
        result["flags"].sort(
            key=lambda item: (item["code"], item["message_indices"], item["evidence"])
        )
        result["decision"] = _decision(result["flags"])
        decision_counts[result["decision"]] += 1
        code_counts.update(flag["code"] for flag in result["flags"])
    corpus_flags.sort(key=lambda item: (item["code"], item["record_ids"], item["evidence"]))
    return {
        "gate_version": GATE_VERSION,
        "policy": {
            "eval_overlap_words": eval_overlap_words,
            "verbatim_span_words": verbatim_span_words,
            "opening_words": opening_words,
            "opening_repeat_min": opening_repeat_min,
            "near_duplicate_threshold": near_duplicate_threshold,
        },
        "summary": {
            "records_scanned": len(records),
            "pass": decision_counts["pass"],
            "review": decision_counts["review"],
            "block": decision_counts["block"],
            "flag_counts": dict(sorted(code_counts.items())),
            "corpus_flag_count": len(corpus_flags),
        },
        "records": results,
        "corpus_flags": corpus_flags,
    }


def read_jsonl(path: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    records: list[Any] = []
    parse_flags: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append(None)
                parse_flags.append(
                    {
                        "line": line_number,
                        "code": "invalid_json",
                        "severity": "block",
                        "message": str(exc),
                    }
                )
    return records, parse_flags


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Candidate JSONL")
    parser.add_argument("--output", type=Path, help="Report JSON; stdout when omitted")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--eval-source",
        type=Path,
        action="append",
        help="Evaluation JSON to scan for phrase overlap; repeatable. Defaults to held-out manifest sources.",
    )
    parser.add_argument("--eval-overlap-words", type=int, default=8)
    parser.add_argument("--verbatim-span-words", type=int, default=16)
    parser.add_argument("--opening-words", type=int, default=8)
    parser.add_argument("--opening-repeat-min", type=int, default=3)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.82)
    parser.add_argument(
        "--fail-on",
        choices=("never", "block", "review"),
        default="block",
        help="Exit 1 on blockers (default), on any flag, or never.",
    )
    args = parser.parse_args(argv)

    for name in ("eval_overlap_words", "verbatim_span_words", "opening_words"):
        if getattr(args, name) < 3:
            parser.error(f"--{name.replace('_', '-')} must be at least 3")
    if args.opening_repeat_min < 2:
        parser.error("--opening-repeat-min must be at least 2")
    if not 0.0 < args.near_duplicate_threshold <= 1.0:
        parser.error("--near-duplicate-threshold must be in (0, 1]")

    records, parse_flags = read_jsonl(args.input)
    eval_paths = args.eval_source or default_eval_paths(args.repo_root)
    references = load_eval_references(eval_paths)
    report = analyze_records(
        records,
        references,
        eval_overlap_words=args.eval_overlap_words,
        verbatim_span_words=args.verbatim_span_words,
        opening_words=args.opening_words,
        opening_repeat_min=args.opening_repeat_min,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    report["input"] = str(args.input)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    report["eval_sources"] = [str(path) for path in eval_paths]
    report["parse_flags"] = parse_flags
    if parse_flags:
        report["summary"]["block"] += len(parse_flags)
        report["summary"]["flag_counts"]["invalid_json"] = len(parse_flags)

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)

    has_block = bool(report["summary"]["block"])
    has_review = bool(report["summary"]["review"] or has_block)
    if args.fail_on == "block" and has_block:
        return 1
    if args.fail_on == "review" and has_review:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
