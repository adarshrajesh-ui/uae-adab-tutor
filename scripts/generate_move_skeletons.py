#!/usr/bin/env python3
"""Generate deterministic UAE Adab lesson skeletons without dialogue text.

The output is a symbolic plan for later realization. It intentionally contains
no student or tutor utterances, so it can be audited for coverage and leakage
before any model is asked to write prose.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAMMAR = PROJECT_ROOT / "data" / "uae_adab" / "move_grammar.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "uae_adab" / "skeletons_v1.jsonl"

LEARNER_CONTEXTS = [
    "homework_review",
    "class_preparation",
    "project_planning",
    "revision_after_feedback",
    "assessment_preparation",
    "independent_study",
]

LEARNER_VOICES = [
    "hesitant_but_engaged",
    "rushed_and_answer_seeking",
    "confident_but_mistaken",
    "discouraged_after_an_error",
    "skeptical_and_evidence_seeking",
    "curious_and_reflective",
]

RESPONSE_CADENCES = [
    "brief_then_expand_if_needed",
    "question_hint_check",
    "worked_micro_step_then_transfer",
    "compare_options_then_justify",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_number(seed: int, key: str) -> int:
    payload = f"{seed}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_choice(items: list[Any], seed: int, key: str) -> Any:
    if not items:
        raise ValueError(f"Cannot choose from an empty list for {key}")
    return items[stable_number(seed, key) % len(items)]


def is_non_monotonic(values: list[int]) -> bool:
    deltas = [right - left for left, right in zip(values, values[1:])]
    return any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas)


def validate_grammar(grammar: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "generator_seed",
        "dataset_split",
        "held_out_eval",
        "registers",
        "length_profiles",
        "phase_sequence",
        "correction_dial",
        "move_correction_levels",
        "global_invariants",
        "move_requirements",
        "scenario_families",
        "subjects",
    }
    missing = required - grammar.keys()
    if missing:
        raise ValueError(f"Grammar is missing keys: {sorted(missing)}")

    phases = grammar["phase_sequence"]
    if len(phases) != 10 or len(set(phases)) != 10:
        raise ValueError("phase_sequence must contain ten unique phases")

    move_requirements = grammar["move_requirements"]
    correction_levels = grammar["move_correction_levels"]
    if set(move_requirements) != set(correction_levels):
        raise ValueError("Every tutor move must have requirements and a correction level")

    for length_name, profile in grammar["length_profiles"].items():
        count = profile["round_count"]
        indexes = profile["blueprint_indexes"]
        if count != len(indexes) or len(set(indexes)) != count:
            raise ValueError(f"Invalid blueprint indexes for {length_name}")
        if min(indexes) < 0 or max(indexes) >= len(phases):
            raise ValueError(f"Blueprint index out of range for {length_name}")
        for pattern in profile["pressure_patterns"]:
            if len(pattern) != count:
                raise ValueError(f"Pressure pattern length mismatch for {length_name}")
            if any(not isinstance(value, int) or not 0 <= value <= 5 for value in pattern):
                raise ValueError(f"Pressure values must be integers from 0 to 5 for {length_name}")
            if not is_non_monotonic(pattern):
                raise ValueError(f"Pressure pattern must rise and fall for {length_name}")

    for family_name, family in grammar["scenario_families"].items():
        for field in ("student_moves", "tutor_moves", "pressure_kinds"):
            if len(family[field]) != len(phases):
                raise ValueError(f"{family_name}.{field} must have ten entries")
        unknown = set(family["tutor_moves"]) - move_requirements.keys()
        if unknown:
            raise ValueError(f"{family_name} uses unknown tutor moves: {sorted(unknown)}")

    for subject_name, subject in grammar["subjects"].items():
        if not subject.get("tasks"):
            raise ValueError(f"Subject {subject_name} has no tasks")
        task_ids = [task["id"] for task in subject["tasks"]]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"Subject {subject_name} has duplicate task IDs")


def load_reserved_eval_metadata(
    grammar: dict[str, Any], project_root: Path = PROJECT_ROOT
) -> tuple[set[str], set[str], list[str]]:
    """Read only IDs and family names; eval turns never enter generation context."""

    reserved_ids: set[str] = set()
    reserved_families: set[str] = set()
    checked_paths: list[str] = []
    for relative in grammar["held_out_eval"]["paths"]:
        path = project_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Held-out eval file is missing: {path}")
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Held-out eval file must contain a list: {path}")
        for scenario in payload:
            reserved_ids.add(str(scenario["id"]))
            reserved_families.add(str(scenario["family"]))
        checked_paths.append(relative)
    return reserved_ids, reserved_families, checked_paths


def pressure_direction(pattern: list[int], position: int) -> str:
    if position == 0:
        return "initial"
    if pattern[position] > pattern[position - 1]:
        return "rising"
    if pattern[position] < pattern[position - 1]:
        return "falling"
    return "steady"


def effective_pressure_kind(base_kind: str, level: int) -> str:
    if base_kind == "none" and level > 0:
        return "background_time_or_confusion_pressure"
    return base_kind


def state_after_move(
    tutor_move: str, pressure_kind: str, previous: dict[str, str]
) -> dict[str, str]:
    state = dict(previous)
    if tutor_move == "specific_face_safe_correction":
        state["misconception"] = "identified_and_under_revision"
    elif tutor_move in {"invite_self_check", "verify_understanding"}:
        state["misconception"] = "learner_verification_pending"
    elif tutor_move == "transfer_student_ownership":
        state["misconception"] = "transfer_check_pending"
    elif tutor_move == "consolidate_method_and_conduct":
        state["misconception"] = "method_consolidated"

    if any(token in pressure_kind for token in ("drift", "drop_style", "generic_tutor")):
        state["drift"] = "probe_detected"
    if tutor_move == "reanchor_to_learning_goal_and_register":
        state["drift"] = "reanchored"
    if tutor_move == "protect_authorship_offer_parallel_practice":
        state["integrity"] = "boundary_held_learning_continues"
    if tutor_move == "state_scope_boundary_continue_academic_help":
        state["religious_boundary"] = "boundary_held_academic_scope_continues"
    if tutor_move == "model_respectful_evidence_question":
        state["authority"] = "respect_and_truth_seeking_balanced"
    return state


def make_stable_id(
    version: str,
    seed: int,
    subject_code: str,
    family_code: str,
    length_code: str,
    register_code: str,
    task_id: str,
) -> str:
    key = ":".join(
        [version, str(seed), subject_code, family_code, length_code, register_code, task_id]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    major = version.split(".", 1)[0]
    return (
        f"uag{major}-{subject_code}-{family_code}-{length_code}-"
        f"{register_code}-{digest}"
    )


def build_skeleton(
    grammar: dict[str, Any],
    seed: int,
    subject_name: str,
    family_name: str,
    length_name: str,
    register_name: str,
    reserved_ids: set[str],
    reserved_families: set[str],
    checked_paths: list[str],
) -> dict[str, Any]:
    subject = grammar["subjects"][subject_name]
    family = grammar["scenario_families"][family_name]
    length = grammar["length_profiles"][length_name]
    register = grammar["registers"][register_name]

    twin_key = ":".join([subject_name, family_name, length_name])
    combination_key = f"{twin_key}:{register_name}"
    task = stable_choice(subject["tasks"], seed, f"task:{twin_key}")
    pattern = stable_choice(
        length["pressure_patterns"], seed, f"pressure:{twin_key}:{task['id']}"
    )
    realization_variant = {
        "learner_context": stable_choice(
            LEARNER_CONTEXTS, seed, f"context:{twin_key}:{task['id']}"
        ),
        "learner_voice": stable_choice(
            LEARNER_VOICES, seed, f"voice:{twin_key}:{task['id']}"
        ),
        "response_cadence": stable_choice(
            RESPONSE_CADENCES, seed, f"cadence:{twin_key}:{task['id']}"
        ),
        "wording_constraint": (
            "Use original natural wording; do not copy the learning objective or "
            "misconception text verbatim into the dialogue."
        ),
    }
    skeleton_id = make_stable_id(
        grammar["schema_version"],
        seed,
        subject["code"],
        family["code"],
        length["code"],
        register["code"],
        task["id"],
    )
    if skeleton_id in reserved_ids:
        raise ValueError(f"Generated ID collides with held-out eval: {skeleton_id}")
    if family_name in reserved_families:
        raise ValueError(f"Generated family collides with held-out eval: {family_name}")

    current_state = {
        "authority": "respect_with_evidence",
        "drift": "anchored",
        "integrity": "student_ownership_expected",
        "misconception": "unassessed",
        "religious_boundary": "within_academic_scope",
    }
    rounds: list[dict[str, Any]] = []
    for position, blueprint_index in enumerate(length["blueprint_indexes"]):
        tutor_move = family["tutor_moves"][blueprint_index]
        level = pattern[position]
        pressure_kind = effective_pressure_kind(
            family["pressure_kinds"][blueprint_index], level
        )
        before = dict(current_state)
        current_state = state_after_move(tutor_move, pressure_kind, current_state)
        rounds.append(
            {
                "round_id": f"r{position + 1:02d}",
                "phase": grammar["phase_sequence"][blueprint_index],
                "pressure": {
                    "direction": pressure_direction(pattern, position),
                    "kind": pressure_kind,
                    "level": level,
                },
                "student_move": {
                    "act": family["student_moves"][blueprint_index],
                    "realization": "not_generated",
                },
                "tutor_plan": {
                    "authority_stance": "respect_without_infallibility",
                    "boundary_stance": "academic_help_continues_within_scope",
                    "correction_level": grammar["move_correction_levels"][tutor_move],
                    "integrity_stance": "preserve_student_authorship",
                    "invariant_checks": sorted(grammar["global_invariants"]),
                    "primary_move": tutor_move,
                    "register_action": register_name,
                    "requirements": grammar["move_requirements"][tutor_move],
                    "realization": "not_generated",
                },
                "state_transition": {"before": before, "after": dict(current_state)},
            }
        )

    return {
        "id": skeleton_id,
        "schema_version": grammar["schema_version"],
        "split": grammar["dataset_split"],
        "source_type": "symbolic_synthetic_plan",
        "generator_seed": seed,
        "design_key": combination_key,
        "register_twin_key": twin_key,
        "subject": subject_name,
        "academic_task": dict(task),
        "realization_variant": realization_variant,
        "register": {
            "name": register_name,
            "deen_explicitness": register["deen_explicitness"],
            "required": register["required"],
            "forbidden": register["forbidden"],
        },
        "length": {"name": length_name, "round_count": length["round_count"]},
        "scenario_family": family_name,
        "capabilities": family["capabilities"],
        "global_invariants": grammar["global_invariants"],
        "initial_lesson_state": {
            "learning_goal": task["objective"],
            "known_misconception": task["misconception"],
            **rounds[0]["state_transition"]["before"],
        },
        "pressure_profile": {
            "levels": pattern,
            "non_monotonic": is_non_monotonic(pattern),
            "scale": "0_no_pressure_to_5_maximum_pressure",
        },
        "rounds": rounds,
        "completion_criteria": [
            "academic_method_is_correct_and_transferable",
            "misconception_is_reduced_or_explicitly_diagnosed",
            "learner_retains_authorship_and_next_step",
            "dignity_authority_integrity_and_boundary_invariants_hold",
            "register_holds_through_final_round",
        ],
        "held_out_exclusion": {
            "checked_against": checked_paths,
            "eval_family_overlap": False,
            "eval_id_overlap": False,
            "eval_turns_used_as_generation_seed": False,
            "status": "passed_metadata_only_exclusion",
        },
        "dialogue_realization": "absent_by_design",
    }


def generate_skeletons(
    grammar: dict[str, Any],
    seed: int,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    validate_grammar(grammar)
    reserved_ids, reserved_families, checked_paths = load_reserved_eval_metadata(
        grammar, project_root
    )
    combinations: Iterable[tuple[str, str, str, str]] = itertools.product(
        grammar["subjects"],
        grammar["scenario_families"],
        grammar["length_profiles"],
        grammar["registers"],
    )
    skeletons = [
        build_skeleton(
            grammar,
            seed,
            subject_name,
            family_name,
            length_name,
            register_name,
            reserved_ids,
            reserved_families,
            checked_paths,
        )
        for subject_name, family_name, length_name, register_name in combinations
    ]
    ids = [item["id"] for item in skeletons]
    if len(ids) != len(set(ids)):
        raise ValueError("Stable ID collision detected")
    return skeletons


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", type=Path, default=DEFAULT_GRAMMAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--variant-count",
        type=int,
        default=1,
        help="Number of full balanced grids to emit using consecutive seeds.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Optional deterministic prefix size; default emits the full balanced grid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grammar = read_json(args.grammar)
    seed = grammar["generator_seed"] if args.seed is None else args.seed
    if args.variant_count <= 0:
        raise ValueError("--variant-count must be positive")
    skeletons = []
    for variant_seed in range(seed, seed + args.variant_count):
        skeletons.extend(generate_skeletons(grammar, variant_seed))
    ids = [item["id"] for item in skeletons]
    if len(ids) != len(set(ids)):
        raise ValueError("Stable ID collision detected across seed variants")
    if args.count is not None:
        if args.count <= 0 or args.count > len(skeletons):
            raise ValueError(f"--count must be between 1 and {len(skeletons)}")
        skeletons = skeletons[: args.count]
    write_jsonl(args.output, skeletons)

    counts = {
        "families": len({row["scenario_family"] for row in skeletons}),
        "lengths": len({row["length"]["name"] for row in skeletons}),
        "registers": len({row["register"]["name"] for row in skeletons}),
        "skeletons": len(skeletons),
        "subjects": len({row["subject"] for row in skeletons}),
    }
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
