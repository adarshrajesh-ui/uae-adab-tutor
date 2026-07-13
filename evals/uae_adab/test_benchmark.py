from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "uae_adab_benchmark", HERE / "run_benchmark.py"
)
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def perfect_judgment() -> dict:
    return {
        "scores": {dimension: 2 for dimension in BENCHMARK.DIMENSIONS},
        "hard_gates": {gate: False for gate in BENCHMARK.HARD_GATES},
        "delete_respect_sentence": {
            "applicable": False,
            "candidate_sentence": None,
            "explanation_unchanged_after_deletion": False,
            "reason": "The method is integrated.",
        },
        "rationale": "Full pass.",
    }


class BenchmarkTests(unittest.TestCase):
    def test_scenario_contract(self) -> None:
        scenarios = BENCHMARK.validate_scenarios(
            json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))
        )
        self.assertEqual(len(scenarios), 10)
        self.assertEqual(len({scenario["id"] for scenario in scenarios}), 10)
        self.assertTrue(all(len(scenario["turns"]) == 5 for scenario in scenarios))

    def test_authentic_heldout_scenario_contract(self) -> None:
        scenarios = BENCHMARK.validate_scenarios(
            json.loads(
                (HERE / "authentic_heldout_scenarios.json").read_text(encoding="utf-8")
            )
        )
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]["id"], "group_leader_service_not_status")
        self.assertEqual(len(scenarios[0]["turns"]), 5)

    def test_source_transfer_scenario_contract(self) -> None:
        scenarios = BENCHMARK.validate_scenarios(
            json.loads(
                (HERE / "source_transfer_scenarios_v0.json").read_text(encoding="utf-8")
            )
        )
        self.assertEqual(len(scenarios), 20)
        self.assertEqual(len({scenario["id"] for scenario in scenarios}), 20)
        self.assertTrue(all(len(scenario["turns"]) == 5 for scenario in scenarios))

    def test_perfect_judgment_is_strict_pass(self) -> None:
        normalized = BENCHMARK.normalize_judgment(perfect_judgment())
        self.assertTrue(normalized["strict_pass"])
        self.assertEqual(normalized["score_total"], 10)

    def test_decorative_respect_fails_strict_pass(self) -> None:
        raw = perfect_judgment()
        raw["delete_respect_sentence"] = {
            "applicable": True,
            "candidate_sentence": "Please be respectful.",
            "explanation_unchanged_after_deletion": True,
            "reason": "All teaching remains generic after deletion.",
        }
        normalized = BENCHMARK.normalize_judgment(raw)
        self.assertFalse(normalized["strict_pass"])
        self.assertFalse(normalized["delete_respect_sentence"]["pass"])

    def test_any_hard_gate_fails_strict_pass(self) -> None:
        raw = perfect_judgment()
        raw["hard_gates"]["humiliation_or_contempt"] = True
        normalized = BENCHMARK.normalize_judgment(raw)
        self.assertFalse(normalized["strict_pass"])
        self.assertFalse(normalized["hard_gate_pass"])

    def test_resume_rejects_checkpoint_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "results.jsonl"
            checkpoint.write_text('{"status":"success"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without a matching manifest"):
                BENCHMARK.validate_resume_state(
                    root / "manifest.json", checkpoint, "current-signature"
                )

    def test_resume_rejects_mismatched_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"run_signature_sha256": "old-signature"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different prompt"):
                BENCHMARK.validate_resume_state(
                    manifest, root / "results.jsonl", "current-signature"
                )

    def test_resume_accepts_matching_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"run_signature_sha256": "current-signature"}),
                encoding="utf-8",
            )
            BENCHMARK.validate_resume_state(
                manifest, root / "results.jsonl", "current-signature"
            )


if __name__ == "__main__":
    unittest.main()
