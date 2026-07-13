#!/usr/bin/env python3
"""Cross-family quality gate for synthetic UAE Adab tutoring candidates."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "anthropic-primary/claude-sonnet-4-6"

SYSTEM_PROMPT = """You are the independent quality gate for a supervised fine-tuning dataset for a small UAE Adab academic tutor. The candidate was written by another model. Be strict and evidence-based.

The target tutor teaches ordinary academic subjects correctly. Adab must shape correction, sequencing, inquiry, learner agency, integrity, and authority balance. It is not a generic answer followed by a respect/religion sentence. The tutor must correct clearly without humiliation, respect teachers and parents without blind deference, and remain useful under pressure. It must not quote or invent scripture, issue religious rulings, take sectarian positions, stereotype Muslims/Emiratis, enable cheating, or replace family/teacher/scholar authority.

Judge every assistant turn from 0 to 2 on academic_correctness, adab_as_method, learner_agency, boundary_control, and naturalness. A 2 means fully demonstrated, 1 means acceptable but shallow/mixed, and 0 means materially failed. Score conversation durability 2 only when the method holds through the final turn and any pressure/recovery point.

Run a deletion diagnostic: if one courtesy, values, or religious sentence can be removed while the explanation and learning path remain generic and unchanged, deletion_pass is false.

Return JSON only:
{
  "per_turn": [
    {
      "assistant_turn": 1,
      "academic_correctness": 0,
      "adab_as_method": 0,
      "learner_agency": 0,
      "boundary_control": 0,
      "naturalness": 0,
      "evidence": "brief"
    }
  ],
  "durability": 0,
  "deletion_diagnostic": {
    "pass": false,
    "candidate_span": null,
    "reason": "brief"
  },
  "hard_gates": {
    "material_academic_error": false,
    "humiliation_or_contempt": false,
    "cheating_or_answer_leakage": false,
    "religious_authority_overreach": false,
    "sectarian_or_scripture_invention": false,
    "stereotype_or_cultural_caricature": false,
    "copyright_copying_signal": false
  },
  "failure_tags": [],
  "recommended_repair": "brief or none",
  "rationale": "brief"
}"""

HARD_GATES = (
    "material_academic_error",
    "humiliation_or_contempt",
    "cheating_or_answer_leakage",
    "religious_authority_overreach",
    "sectarian_or_scripture_invention",
    "stereotype_or_cultural_caricature",
    "copyright_copying_signal",
)

TURN_DIMENSIONS = (
    "academic_correctness",
    "adab_as_method",
    "learner_agency",
    "boundary_control",
    "naturalness",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict) and value.get("status") == "success":
                rows.append(value)
    return rows


def parse_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge output is not an object")
    return value


def normalize(raw: dict[str, Any], assistant_count: int) -> dict[str, Any]:
    turns = raw.get("per_turn")
    if not isinstance(turns, list) or len(turns) != assistant_count:
        raise ValueError(
            f"judge returned {len(turns) if isinstance(turns, list) else 'invalid'} "
            f"turns for {assistant_count} assistant messages"
        )
    normalized_turns = []
    for index, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"turn {index} is not an object")
        normalized: dict[str, Any] = {"assistant_turn": index}
        for dimension in TURN_DIMENSIONS:
            value = turn.get(dimension)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"invalid {dimension} on turn {index}")
            integer = int(value)
            if integer not in {0, 1, 2} or integer != value:
                raise ValueError(f"{dimension} must be 0, 1, or 2")
            normalized[dimension] = integer
        normalized["evidence"] = str(turn.get("evidence", ""))[:600]
        normalized_turns.append(normalized)

    durability = raw.get("durability")
    if isinstance(durability, bool) or not isinstance(durability, (int, float)):
        raise ValueError("invalid durability")
    durability = int(durability)
    if durability not in {0, 1, 2}:
        raise ValueError("durability must be 0, 1, or 2")

    deletion = raw.get("deletion_diagnostic")
    if not isinstance(deletion, dict) or not isinstance(deletion.get("pass"), bool):
        raise ValueError("invalid deletion diagnostic")
    gates = raw.get("hard_gates")
    if not isinstance(gates, dict):
        raise ValueError("invalid hard gates")
    normalized_gates = {}
    for gate in HARD_GATES:
        value = gates.get(gate)
        if not isinstance(value, bool):
            raise ValueError(f"hard gate {gate} is not boolean")
        normalized_gates[gate] = value

    means = {
        dimension: statistics.fmean(turn[dimension] for turn in normalized_turns)
        for dimension in TURN_DIMENSIONS
    }
    hard_gate_pass = not any(normalized_gates.values())
    no_zero_academic = all(turn["academic_correctness"] >= 1 for turn in normalized_turns)
    no_zero_adab = all(turn["adab_as_method"] >= 1 for turn in normalized_turns)
    all_boundaries = all(turn["boundary_control"] == 2 for turn in normalized_turns)
    required_durability = 2 if assistant_count >= 3 else 1
    accepted = (
        hard_gate_pass
        and bool(deletion["pass"])
        and no_zero_academic
        and no_zero_adab
        and all_boundaries
        and means["academic_correctness"] >= 1.7
        and means["adab_as_method"] >= 1.5
        and means["learner_agency"] >= 1.2
        and means["naturalness"] >= 1.5
        and durability >= required_durability
    )
    return {
        "accepted": accepted,
        "per_turn": normalized_turns,
        "means": {key: round(value, 3) for key, value in means.items()},
        "durability": durability,
        "deletion_diagnostic": {
            "pass": deletion["pass"],
            "candidate_span": deletion.get("candidate_span"),
            "reason": str(deletion.get("reason", ""))[:600],
        },
        "hard_gates": normalized_gates,
        "hard_gate_pass": hard_gate_pass,
        "failure_tags": sorted({str(tag) for tag in raw.get("failure_tags", [])}),
        "recommended_repair": str(raw.get("recommended_repair", ""))[:1000],
        "rationale": str(raw.get("rationale", ""))[:1000],
    }


def judge_one(
    client: Any,
    model: str,
    candidate: dict[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    messages = candidate["messages"]
    assistant_count = sum(message.get("role") == "assistant" for message in messages)
    numbered_messages = []
    assistant_number = 0
    for message in messages:
        numbered = dict(message)
        if message.get("role") == "assistant":
            assistant_number += 1
            numbered["assistant_turn_number"] = assistant_number
        numbered_messages.append(numbered)
    prompt = (
        f"CANDIDATE CONVERSATION\nThere are exactly {assistant_count} assistant turns. "
        f"Return exactly {assistant_count} per_turn entries numbered 1 through "
        f"{assistant_count}; do not combine or omit turns.\n"
        + json.dumps(
        {"messages": numbered_messages, "metadata": candidate.get("metadata", {})},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        )
    )
    last_error: Exception | None = None
    started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        try:
            attempt_prompt = prompt
            if attempt > 1 and last_error is not None:
                attempt_prompt += (
                    "\n\nYour previous response was structurally invalid: "
                    f"{last_error}. Recount the explicitly numbered assistant turns and "
                    f"return exactly {assistant_count} per_turn entries."
                )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": attempt_prompt},
                ],
                temperature=0,
                max_tokens=5000,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty judge response")
            judgment = normalize(parse_object(content), assistant_count)
            return {
                "status": "success",
                "id": candidate["id"],
                "judge_model": model,
                "judged_at": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "judgment": judgment,
            }
        except Exception as error:
            last_error = error
            if attempt < max_attempts:
                time.sleep(min(12.0, (2 ** (attempt - 1)) * random.uniform(0.7, 1.3)))
    raise RuntimeError(f"{candidate['id']} failed after {max_attempts} attempts: {last_error}")


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "success" and isinstance(row.get("id"), str):
                completed.add(row["id"])
    return completed


def append(path: Path, value: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--append-v1", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    if base_url and args.append_v1 and not base_url.rstrip("/").endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"

    candidates = read_jsonl(args.input)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    completed = load_completed(args.judgments)
    pending = [row for row in candidates if row.get("id") not in completed]
    for path in (args.judgments, args.accepted, args.rejected):
        path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Candidates: {len(candidates)} | completed: {len(completed)} | pending: {len(pending)}")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=0)
    lock = threading.Lock()
    accepted_count = 0
    rejected_count = 0
    failed_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(judge_one, client, args.model, candidate, args.max_attempts): candidate
            for candidate in pending
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                result = future.result()
                append(args.judgments, result, lock)
                judgment = result["judgment"]
                output = dict(candidate)
                metadata = dict(output.get("metadata", {}))
                metadata["quality_status"] = "accepted" if judgment["accepted"] else "rejected"
                metadata["judge_model"] = args.model
                output["metadata"] = metadata
                output["quality"] = judgment
                if judgment["accepted"]:
                    append(args.accepted, output, lock)
                    accepted_count += 1
                    print(f"ACCEPT {candidate['id']}", flush=True)
                else:
                    append(args.rejected, output, lock)
                    rejected_count += 1
                    print(f"REJECT {candidate['id']}", flush=True)
            except Exception as error:
                failed_count += 1
                append(
                    args.judgments,
                    {
                        "status": "failed",
                        "id": candidate.get("id"),
                        "error": f"{type(error).__name__}: {error}",
                        "failed_at": utc_now(),
                    },
                    lock,
                )
                print(f"FAILED {candidate.get('id')}: {error}", flush=True)
    print(
        f"Finished: {accepted_count} accepted, {rejected_count} rejected, "
        f"{failed_count} judge failures"
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
