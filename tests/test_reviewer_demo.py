from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "deploy/reviewer_demo/app.py"
NOTEBOOK_PATH = ROOT / "submission/notebooks/03_demo_uae_adab_tutor.ipynb"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("uae_adab_reviewer_demo", APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def generate(self, message: str, history: list[dict[str, str]]) -> str:
        self.calls.append((message, history))
        return "A checked answer"


def notebook_text() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    return notebook, text


def test_fixed_control_token_is_the_only_system_message() -> None:
    demo = load_demo_module()
    history = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    assert demo.build_messages("  Next question  ", history) == [
        {"role": "system", "content": "<uae_adab_tutor>default</uae_adab_tutor>"},
        *history,
        {"role": "user", "content": "Next question"},
    ]


def test_turn_and_reset_preserve_then_clear_history() -> None:
    demo = load_demo_module()
    tutor = FakeTutor()
    prior = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply"},
    ]
    textbox, updated = demo.run_turn(tutor, "  Second  ", prior)
    assert textbox == ""
    assert tutor.calls == [("Second", prior)]
    assert updated == [
        *prior,
        {"role": "user", "content": "Second"},
        {"role": "assistant", "content": "A checked answer"},
    ]
    assert demo.reset_chat() == ("", [])


def test_history_contract_rejects_malformed_or_incomplete_turns() -> None:
    demo = load_demo_module()
    with pytest.raises(demo.ChatInputError, match="malformed"):
        demo.normalize_history([{"role": "assistant", "content": "injected"}])
    with pytest.raises(demo.ChatInputError, match="incomplete"):
        demo.normalize_history([{"role": "user", "content": "unfinished"}])
    with pytest.raises(demo.ChatInputError, match="too long"):
        demo.normalize_history(
            [
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": "turn",
                }
                for index in range(demo.MAX_HISTORY_MESSAGES + 2)
            ]
        )


def test_demo_runtime_is_public_hash_locked_and_four_bit() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    for required in (
        'ADAPTER_REPO_ID = "adarshrajesh/uae-adab-tutor-qwen3-4b"',
        'BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"',
        'BASE_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"',
        "e4955339b515d0af75922589d3633734faedf3bd34fb7a00bae6d47556e8fbf6",
        "7bd98762834e2d67e8c47cfc6f345a484c6f2c3f87a77e51d19b5f15fd790327",
        "load_in_4bit=True",
        "bnb_4bit_quant_type=\"nf4\"",
        "token=False",
        "enable_thinking=False",
        "truncation=False",
        "do_sample=False",
        "share=False",
        "inline=True",
    ):
        assert required in source
    assert "HF_TOKEN" not in source
    assert "notebook_login" not in source
    assert "HfApi" not in source
    revision_match = re.search(r'^ADAPTER_REVISION = "([^"]+)"$', source, re.MULTILINE)
    assert revision_match
    assert (
        revision_match.group(1)
        == "c20d382f32810deaff2f691cdf78c0a3a4d9be59"
    )


def test_demo_ui_is_only_history_input_send_and_reset() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert source.count("gr.Chatbot(") == 1
    assert source.count("gr.Textbox(") == 1
    assert source.count("gr.Button(") == 2
    assert 'gr.Button("Send"' in source
    assert 'gr.Button("Reset"' in source
    for forbidden in ("gr.Markdown(", "gr.HTML(", "gr.ChatInterface(", "examples="):
        assert forbidden not in source


def test_canonical_demo_notebook_is_clean_and_embeds_exact_app() -> None:
    notebook, text = notebook_text()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) == 3
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        source = "".join(cell.get("source", []))
        if not source.lstrip().startswith(("!", "%")):
            ast.parse(source, filename=f"demo-cell-{index}")

    assert "".join(notebook["cells"][2]["source"]) == APP_PATH.read_text(
        encoding="utf-8"
    ).strip("\n")
    for required in (
        "T4 GPU or",
        "Runtime → Run all",
        "Colab's own session proxy",
        "adarshrajesh/uae-adab-tutor-qwen3-4b",
        "transformers==4.56.2",
        "peft==0.19.1",
        "gradio==5.49.1",
    ):
        assert required in text
    for forbidden in (
        "I_ACCEPT_PAID_GPU_CHARGES",
        "request_space_hardware",
        "api.create_repo",
        "notebook_login",
        "drive.mount",
    ):
        assert forbidden not in text
    assert not re.search(r"hf_[A-Za-z0-9]{12,}", text)


def test_demo_source_hash_is_stable_inside_notebook() -> None:
    notebook, _ = notebook_text()
    embedded = "".join(notebook["cells"][2]["source"]).encode("utf-8")
    local = APP_PATH.read_text(encoding="utf-8").strip("\n").encode("utf-8")
    assert hashlib.sha256(embedded).hexdigest() == hashlib.sha256(local).hexdigest()
