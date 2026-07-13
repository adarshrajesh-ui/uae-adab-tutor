# Run the temporary Colab demo

The submission uses a temporary Google Colab session for live inference.

Open the demo here:

<https://colab.research.google.com/github/adarshrajesh-ui/uae-adab-tutor/blob/main/submission/notebooks/03_demo_uae_adab_tutor.ipynb>

## What the demo loads

- Base: `Qwen/Qwen3-4B-Instruct-2507`
- Base revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Adapter: `adarshrajesh/uae-adab-tutor-qwen3-4b`
- System token: `<uae_adab_tutor>default</uae_adab_tutor>`
- Native Qwen template with thinking disabled
- 4-bit inference on an L4 or A100

The expected original adapter fingerprint is
`c98e6ebda3e2bd8f6b885a318f8987035547477f796717c76d4846824df20ce0`.

## Reviewer steps

1. Open the Colab link.
2. Select an L4 or A100 GPU runtime.
3. Run cells from top to bottom and restart only when the install cell asks.
4. Confirm the pinned base and adapter checks pass.
5. Reset the chat before a new scenario.
6. Disconnect the runtime after testing or recording.

The stable URL opens the notebook. The GPU process and live chat exist only
for the active Colab session.

## Three-window video layout

Place these windows side by side:

1. Claude question-only in a fresh temporary chat.
2. Claude with the frozen strong prompt in a separate fresh temporary chat.
3. Fine-tuned Qwen in Colab.

Send identical turns in the same order. Label the Qwen condition “fixed control
token only.” Label both Claude windows “live illustration, not formal eval.”

## Public artifacts

- GitHub: <https://github.com/adarshrajesh-ui/uae-adab-tutor>
- Model: <https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b>
- Dataset: <https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600>
