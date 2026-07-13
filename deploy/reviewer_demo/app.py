"""Minimal temporary Gradio demo for the selected UAE Adab tutor.

The canonical Colab notebook embeds this file verbatim.  It deliberately has
no product landing page, account layer, persistent hosting, or prose behavior
prompt.  The model receives only the fixed control token used during training.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any


ADAPTER_REPO_ID = "adarshrajesh/uae-adab-tutor-qwen3-4b"
ADAPTER_REVISION = "c20d382f32810deaff2f691cdf78c0a3a4d9be59"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
BASE_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SYSTEM_MESSAGE = "<uae_adab_tutor>default</uae_adab_tutor>"
MAX_SEQUENCE_LENGTH = 4096
MAX_NEW_TOKENS = 512
MAX_USER_CHARACTERS = 4000
MAX_HISTORY_MESSAGES = 40
MINIMUM_GPU_MEMORY_GB = 14.0
EXPECTED_ADAPTER_FILES = {
    "adapter_config.json": "e4955339b515d0af75922589d3633734faedf3bd34fb7a00bae6d47556e8fbf6",
    "adapter_model.safetensors": "7bd98762834e2d67e8c47cfc6f345a484c6f2c3f87a77e51d19b5f15fd790327",
}
EXPECTED_ADAPTER_FINGERPRINT = (
    "c98e6ebda3e2bd8f6b885a318f8987035547477f796717c76d4846824df20ce0"
)


class DemoConfigurationError(RuntimeError):
    """The runtime, base model, or adapter does not match the demo contract."""


class ChatInputError(ValueError):
    """A user-facing message or history error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_user_message(message: Any) -> str:
    if not isinstance(message, str):
        raise ChatInputError("Please enter a text question.")
    cleaned = message.strip()
    if not cleaned:
        raise ChatInputError("Please enter a question before sending.")
    if len(cleaned) > MAX_USER_CHARACTERS:
        raise ChatInputError(
            f"Keep the question under {MAX_USER_CHARACTERS:,} characters."
        )
    return cleaned


def normalize_history(
    history: Sequence[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    if history is None:
        return []
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ChatInputError("The chat history is malformed. Press Reset and try again.")
    if len(history) > MAX_HISTORY_MESSAGES:
        raise ChatInputError("This chat is too long. Press Reset to start a new one.")

    normalized: list[dict[str, str]] = []
    expected_role = "user"
    for item in history:
        if not isinstance(item, dict):
            raise ChatInputError("The chat history is malformed. Press Reset and try again.")
        role = item.get("role")
        content = item.get("content")
        if role != expected_role or not isinstance(content, str) or not content.strip():
            raise ChatInputError("The chat history is malformed. Press Reset and try again.")
        normalized.append({"role": role, "content": content.strip()})
        expected_role = "assistant" if expected_role == "user" else "user"
    if normalized and normalized[-1]["role"] != "assistant":
        raise ChatInputError("The chat history is incomplete. Press Reset and try again.")
    return normalized


def build_messages(
    message: Any,
    history: Sequence[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Build a prompt containing only the learned control token and chat turns."""

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        *normalize_history(history),
        {"role": "user", "content": clean_user_message(message)},
    ]


def download_and_verify_adapter() -> Path:
    """Download the public adapter without credentials and verify exact bytes."""

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        directory = Path(
            snapshot_download(
                repo_id=ADAPTER_REPO_ID,
                revision=ADAPTER_REVISION,
                token=False,
                allow_patterns=sorted(EXPECTED_ADAPTER_FILES),
            )
        )
    except HfHubHTTPError as exc:
        raise DemoConfigurationError(
            f"Could not download public adapter {ADAPTER_REPO_ID}. "
            "Confirm that the repository is public and try again."
        ) from exc

    for filename, expected in EXPECTED_ADAPTER_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise DemoConfigurationError(f"The adapter repository is missing {filename}.")
        if sha256_file(path) != expected:
            raise DemoConfigurationError(f"Adapter hash mismatch: {filename}.")

    config = json.loads(
        (directory / "adapter_config.json").read_text(encoding="utf-8")
    )
    if config.get("base_model_name_or_path") != BASE_MODEL_ID:
        raise DemoConfigurationError("The adapter targets a different base model.")
    if config.get("revision") != BASE_MODEL_REVISION:
        raise DemoConfigurationError("The adapter targets a different base revision.")
    return directory


class QwenAdabTutor:
    """Pinned 4-bit Qwen base with the exact selected PEFT adapter attached."""

    def __init__(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise DemoConfigurationError(
                "Enable a T4 or better GPU in Colab before running the notebook."
            )
        properties = torch.cuda.get_device_properties(0)
        gpu_memory_gb = properties.total_memory / (1024**3)
        if gpu_memory_gb < MINIMUM_GPU_MEMORY_GB:
            raise DemoConfigurationError(
                f"Use a T4-class GPU or better; {properties.name} has "
                f"only {gpu_memory_gb:.1f} GB."
            )

        adapter_directory = download_and_verify_adapter()
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            token=False,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            token=False,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        observed_model_revision = getattr(self.model.config, "_commit_hash", None)
        if observed_model_revision not in {None, BASE_MODEL_REVISION}:
            raise DemoConfigurationError(
                "The base model did not resolve to the pinned revision."
            )
        tokenizer_revision = self.tokenizer.init_kwargs.get("_commit_hash")
        if tokenizer_revision not in {None, BASE_MODEL_REVISION}:
            raise DemoConfigurationError(
                "The tokenizer did not resolve to the pinned revision."
            )

        self.model = PeftModel.from_pretrained(
            self.model,
            str(adapter_directory),
            is_trainable=False,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._torch = torch
        self._generation_lock = threading.Lock()
        print("VERIFIED ADAPTER:", ADAPTER_REPO_ID)
        print("ADAPTER FINGERPRINT:", EXPECTED_ADAPTER_FINGERPRINT)
        print("GPU:", properties.name, f"({gpu_memory_gb:.1f} GB)")

    @property
    def input_device(self) -> Any:
        return self.model.get_input_embeddings().weight.device

    def generate(self, message: Any, history: Any) -> str:
        messages = build_messages(message, history)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self.input_device)
        input_tokens = int(inputs.input_ids.shape[-1])
        if input_tokens + MAX_NEW_TOKENS > MAX_SEQUENCE_LENGTH:
            raise ChatInputError(
                "The 4,096-token context is full. Press Reset to start a new chat."
            )

        with self._generation_lock, self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            output[0][input_tokens:],
            skip_special_tokens=True,
        ).strip()
        if not answer:
            raise RuntimeError("The model returned an empty response. Try again.")
        return answer


def run_turn(
    tutor: Any,
    message: Any,
    history: Sequence[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, str]]]:
    """Generate one response and return a Gradio textbox/history update."""

    cleaned = clean_user_message(message)
    normalized = normalize_history(history)
    answer = tutor.generate(cleaned, normalized)
    return "", [
        *normalized,
        {"role": "user", "content": cleaned},
        {"role": "assistant", "content": answer},
    ]


def reset_chat() -> tuple[str, list[dict[str, str]]]:
    return "", []


def build_demo(tutor: Any) -> Any:
    """Create the intentionally bare four-control reviewer interface."""

    import gradio as gr

    def respond(message: Any, history: Any) -> tuple[str, list[dict[str, str]]]:
        try:
            return run_turn(tutor, message, history)
        except ChatInputError as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(
        title="UAE Adab Tutor",
        analytics_enabled=False,
    ) as demo:
        history = gr.Chatbot(
            label="History",
            type="messages",
            height=560,
        )
        user_input = gr.Textbox(
            label="Input",
            placeholder="Ask a school-subject question…",
            lines=2,
        )
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            reset = gr.Button("Reset")

        send.click(
            respond,
            inputs=[user_input, history],
            outputs=[user_input, history],
            api_name=False,
        )
        user_input.submit(
            respond,
            inputs=[user_input, history],
            outputs=[user_input, history],
            api_name=False,
        )
        reset.click(
            reset_chat,
            outputs=[user_input, history],
            queue=False,
            api_name=False,
        )
    return demo


def main() -> None:
    tutor = QwenAdabTutor()
    demo = build_demo(tutor)
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        share=True,
        debug=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
