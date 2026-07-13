from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("saved_output_scorer", HERE / "score_saved_outputs.py")
assert SPEC and SPEC.loader
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def fixture() -> tuple[list[dict], list[dict]]:
    scenarios = SCORER.BENCHMARK.validate_scenarios(
        json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))
    )
    scenario = scenarios[0]
    row = {
        "schema_version": SCORER.SAVED_OUTPUT_SCHEMA,
        "job_id": f"{scenario['id']}::base_no_prompt",
        "scenario_id": scenario["id"],
        "condition": "base_no_prompt",
        "answer_model": "Qwen/Qwen3-4B-Instruct-2507",
        "adapter": None,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 512,
            "base_model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        },
        "turns": [
            {"turn": index, "user": turn["user"], "answer": f"Answer {index}."}
            for index, turn in enumerate(scenario["turns"], start=1)
        ],
    }
    return [row], scenarios


class SavedOutputScorerTests(unittest.TestCase):
    def test_valid_saved_output_contract(self) -> None:
        rows, scenarios = fixture()
        self.assertEqual(SCORER.validate_saved_outputs(rows, scenarios), rows)

    def test_source_transfer_accepts_all_four_comparison_conditions(self) -> None:
        scenarios = SCORER.BENCHMARK.validate_scenarios(
            json.loads((HERE / "source_transfer_scenarios_v0.json").read_text(encoding="utf-8"))
        )
        conditions = (
            "base_no_prompt",
            "base_strong_prompt",
            "prior_v1_legacy",
            "v2_balanced20_145",
        )
        rows = []
        for scenario in scenarios:
            for condition in conditions:
                rows.append(
                    {
                        "schema_version": SCORER.SAVED_OUTPUT_SCHEMA,
                        "job_id": f"{scenario['id']}::{condition}",
                        "scenario_id": scenario["id"],
                        "condition": condition,
                        "answer_model": "Qwen/Qwen3-4B-Instruct-2507",
                        "adapter": None,
                        "generation_config": {
                            "do_sample": False,
                            "max_new_tokens": 512,
                            "base_model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                        },
                        "turns": [
                            {"turn": index, "user": spec["user"], "answer": f"Placeholder {index}."}
                            for index, spec in enumerate(scenario["turns"], start=1)
                        ],
                    }
                )
        self.assertEqual(len(rows), 80)
        self.assertEqual(SCORER.validate_saved_outputs(rows, scenarios), rows)

    def test_both_frozen_suites_accept_exact_five_condition_matrix(self) -> None:
        conditions = (
            "base_no_prompt",
            "base_strong_prompt",
            "prior_v1_legacy",
            "grounded_120_v2",
            "complete_600_v2",
        )
        for scenario_file, expected_scenarios in (
            ("scenarios.json", 10),
            ("source_transfer_scenarios_v0.json", 20),
        ):
            scenarios = SCORER.BENCHMARK.validate_scenarios(
                json.loads((HERE / scenario_file).read_text(encoding="utf-8"))
            )
            rows = []
            for scenario in scenarios:
                for condition in conditions:
                    rows.append(
                        {
                            "schema_version": SCORER.SAVED_OUTPUT_SCHEMA,
                            "job_id": f"{scenario['id']}::{condition}",
                            "scenario_id": scenario["id"],
                            "condition": condition,
                            "answer_model": "Qwen/Qwen3-4B-Instruct-2507",
                            "adapter": None,
                            "generation_config": {
                                "do_sample": False,
                                "max_new_tokens": 512,
                                "base_model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                            },
                            "turns": [
                                {
                                    "turn": index,
                                    "user": spec["user"],
                                    "answer": f"Placeholder {index}.",
                                }
                                for index, spec in enumerate(scenario["turns"], start=1)
                            ],
                        }
                    )
            self.assertEqual(len(scenarios), expected_scenarios)
            self.assertEqual(len(rows), expected_scenarios * len(conditions))
            self.assertEqual(SCORER.validate_saved_outputs(rows, scenarios), rows)

    def test_changed_heldout_user_turn_is_rejected(self) -> None:
        rows, scenarios = fixture()
        rows[0]["turns"][2]["user"] = "changed"
        with self.assertRaisesRegex(ValueError, "differs from held-out scenario"):
            SCORER.validate_saved_outputs(rows, scenarios)

    def test_duplicate_job_is_rejected(self) -> None:
        rows, scenarios = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate job_id"):
            SCORER.validate_saved_outputs(rows + rows, scenarios)

    def test_nondeterministic_generation_is_rejected(self) -> None:
        rows, scenarios = fixture()
        rows[0]["generation_config"]["do_sample"] = True
        with self.assertRaisesRegex(ValueError, "do_sample=false"):
            SCORER.validate_saved_outputs(rows, scenarios)

    def test_missing_base_revision_is_rejected(self) -> None:
        rows, scenarios = fixture()
        del rows[0]["generation_config"]["base_model_revision"]
        with self.assertRaisesRegex(ValueError, "base_model_revision"):
            SCORER.validate_saved_outputs(rows, scenarios)

    def test_judge_path_is_blind_to_condition_model_and_adapter(self) -> None:
        rows, scenarios = fixture()
        row = rows[0]
        row["adapter"] = "/secret/tuned-adapter"
        observed: list[list[dict]] = []
        perfect = SCORER.BENCHMARK.normalize_judgment(
            {
                "scores": {name: 2 for name in SCORER.BENCHMARK.DIMENSIONS},
                "hard_gates": {name: False for name in SCORER.BENCHMARK.HARD_GATES},
                "delete_respect_sentence": {
                    "applicable": False,
                    "candidate_sentence": None,
                    "explanation_unchanged_after_deletion": False,
                    "reason": "integrated",
                },
                "rationale": "pass",
            }
        )
        original = SCORER.BENCHMARK.judge_once

        def fake_judge(client, args, scenario, turn_spec, messages):
            observed.append([dict(message) for message in messages])
            return perfect

        SCORER.BENCHMARK.judge_once = fake_judge
        try:
            result = SCORER.judge_saved_job(
                None,
                SimpleNamespace(judge_model="independent-judge"),
                row,
                scenarios[0],
            )
        finally:
            SCORER.BENCHMARK.judge_once = original

        visible = json.dumps(observed)
        self.assertNotIn(row["condition"], visible)
        self.assertNotIn(row["answer_model"], visible)
        self.assertNotIn(row["adapter"], visible)
        self.assertEqual(result["condition"], row["condition"])


if __name__ == "__main__":
    unittest.main()
