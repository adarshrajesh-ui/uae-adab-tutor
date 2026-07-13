#!/usr/bin/env python3
"""Blind-judge saved local-model UAE Adab benchmark conversations.

Generation happens in Colab; judging happens later through the configured
OpenAI-compatible gateway. Condition/model fields are never included in the
judge prompt. The output uses the same result and summary contract as
run_benchmark.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("uae_adab_benchmark", HERE / "run_benchmark.py")
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)

SAVED_OUTPUT_SCHEMA = "uae_adab_saved_outputs.v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            rows.append(value)
    return rows


def validate_saved_outputs(
    rows: list[dict[str, Any]], scenarios: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_scenario = {scenario["id"]: scenario for scenario in scenarios}
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != SAVED_OUTPUT_SCHEMA:
            raise ValueError(f"row {index} has unsupported schema_version")
        job_id = row.get("job_id")
        scenario_id = row.get("scenario_id")
        condition = row.get("condition")
        if not all(isinstance(value, str) and value for value in (job_id, scenario_id, condition)):
            raise ValueError(f"row {index} has invalid job identity")
        if job_id in seen:
            raise ValueError(f"duplicate job_id: {job_id}")
        seen.add(job_id)
        if scenario_id not in by_scenario:
            raise ValueError(f"unknown scenario_id: {scenario_id}")
        if job_id != f"{scenario_id}::{condition}":
            raise ValueError(f"job_id does not match scenario and condition: {job_id}")
        turns = row.get("turns")
        scenario_turns = by_scenario[scenario_id]["turns"]
        if not isinstance(turns, list) or len(turns) != len(scenario_turns):
            raise ValueError(f"{job_id} must contain exactly {len(scenario_turns)} turns")
        for turn_number, (turn, spec) in enumerate(zip(turns, scenario_turns), start=1):
            if not isinstance(turn, dict) or turn.get("turn") != turn_number:
                raise ValueError(f"{job_id} has invalid turn numbering")
            if turn.get("user") != spec["user"]:
                raise ValueError(f"{job_id} turn {turn_number} user text differs from held-out scenario")
            if not isinstance(turn.get("answer"), str) or not turn["answer"].strip():
                raise ValueError(f"{job_id} turn {turn_number} has an empty answer")
        config = row.get("generation_config")
        if not isinstance(config, dict) or config.get("do_sample") is not False:
            raise ValueError(f"{job_id} must record deterministic do_sample=false generation")
        if not isinstance(config.get("base_model_revision"), str) or not config["base_model_revision"]:
            raise ValueError(f"{job_id} must record the pinned base_model_revision")
    return rows


def judge_saved_job(client: Any, args: argparse.Namespace, row: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in scenario.get("setup_messages", [])
    ]
    judged_turns: list[dict[str, Any]] = []
    for turn, turn_spec in zip(row["turns"], scenario["turns"]):
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
        judgment = BENCHMARK.judge_once(client, args, scenario, turn_spec, messages)
        judged_turns.append(
            {
                "turn": turn["turn"],
                "user": turn["user"],
                "expected": turn_spec["expected"],
                "fail_if": turn_spec["fail_if"],
                "answer": turn["answer"],
                "judgment": judgment,
                "elapsed_seconds": turn.get("elapsed_seconds"),
            }
        )
    return {
        "status": "success",
        "job_id": row["job_id"],
        "scenario_id": scenario["id"],
        "title": scenario["title"],
        "family": scenario["family"],
        "focus": scenario["focus"],
        "condition": row["condition"],
        "answer_model": row.get("answer_model", "local_model"),
        "judge_model": args.judge_model,
        "turns": judged_turns,
        "metrics": {
            "turn_5_strict_pass": judged_turns[-1]["judgment"]["strict_pass"],
            "all_turn_durable_pass": all(turn["judgment"]["strict_pass"] for turn in judged_turns),
            "strict_turns": sum(turn["judgment"]["strict_pass"] for turn in judged_turns),
            "hard_gate_failures": sum(not turn["judgment"]["hard_gate_pass"] for turn in judged_turns),
            "delete_test_failures": sum(
                not turn["judgment"]["delete_respect_sentence"]["pass"]
                for turn in judged_turns
            ),
        },
        "completed_at": BENCHMARK.utc_now(),
        "generation_config": row.get("generation_config", {}),
        "adapter": row.get("adapter"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="anthropic-primary/claude-sonnet-4-6")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=900)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--append-v1", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenarios = BENCHMARK.validate_scenarios(BENCHMARK.read_json(args.scenarios))
        rows = [row for path in args.input for row in read_jsonl(path)]
        validate_saved_outputs(rows, scenarios)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    conditions = sorted({row["condition"] for row in rows})
    print(f"Validated {len(rows)} saved conversations across conditions: {', '.join(conditions)}")
    if args.dry_run:
        return 0

    results_path = args.output_dir / "results.jsonl"
    if results_path.exists() and results_path.stat().st_size:
        print(
            f"Output error: {results_path} is nonempty; use a fresh --output-dir "
            "to avoid duplicate judgments.",
            file=sys.stderr,
        )
        return 2

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and args.append_v1 and not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.request_timeout, max_retries=0)
    by_scenario = {scenario["id"]: scenario for scenario in scenarios}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(judge_saved_job, client, args, row, by_scenario[row["scenario_id"]]): row
            for row in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(rows)}] {row['job_id']} judged", flush=True)
            except Exception as exc:
                error = {"status": "error", "job_id": row["job_id"], "error": str(exc)}
                errors.append(error)
                print(f"[{index}/{len(rows)}] {row['job_id']}: ERROR {exc}", file=sys.stderr)
                result = error
            BENCHMARK.append_jsonl(results_path, result, lock)

    if results:
        BENCHMARK.write_summaries(args.output_dir, results)
    manifest = {
        "schema_version": SAVED_OUTPUT_SCHEMA,
        "status": "complete" if not errors else "partial",
        "created_at": BENCHMARK.utc_now(),
        "input_files": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in args.input
        ],
        "scenario_sha256": hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
        "conditions": conditions,
        "judge_model": args.judge_model,
        "judge_prompt_sha256": BENCHMARK.sha256_text(BENCHMARK.JUDGE_SYSTEM_PROMPT),
        "successful_jobs": len(results),
        "failed_jobs": len(errors),
        "blind_judge_fields_withheld": ["condition", "answer_model", "adapter", "generation_config"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
